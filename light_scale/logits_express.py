from light_scale.sync_processor import ActorReferenceDataUpdater
from megatron.training.global_vars import get_args
import torch.distributed as dist
import megatron.core.parallel_state as mpu
import os
from light_scale.logger_utils import setup_logger
from light_scale import dist_utils
from light_scale.dist_utils import init_custom_process_group
import torch
from datetime import timedelta

class LogitsExpress:
    def __init__(self, ar_data_updater: ActorReferenceDataUpdater, is_teacher: bool):
        self.ar_data_updater = ar_data_updater
        self.is_teacher = is_teacher
        self.margs = get_args()
        self.logger = setup_logger("light_scale")
        self.logger.info("initializing logits express")
        self.matrix = self.__gather_state_to_rank_matrix()
        self.rank_to_ips = self.__gather_rank_to_ip()
        # Perform cross-world meta exchange and validation (sets self.student_meta or raises)
        self.student_meta, self.teacher_meta = self.__exchange_and_validate_meta()
        self.logits_express_pg_info = self.__find_neighbors_and_create_cross_pg()
        self.logger.info("initialized logits express")

    def __gather_state_to_rank_matrix(self):
        # 通过all gather通信，收集所有rank的state信息到一个矩阵中
        # 返回：matrix[pp_rank][dp_rank][tp_rank] = (global_rank, local_rank), matrix: List[List[Tuple[int, int]]]
        # 每个rank准备自己的信息
        self.logger.info("gathering state to rank matrix")
        my_info = {
            'rank': dist.get_rank(),
            'local_rank': int(os.environ["LOCAL_RANK"]),
            'coords': (mpu.get_pipeline_model_parallel_rank(), mpu.get_data_parallel_rank(), mpu.get_tensor_model_parallel_rank())
        }

        # 准备一个列表来收集所有rank的信息
        all_info = [None] * dist.get_world_size()

        # 使用all_gather_object来收集所有rank的信息
        dist.all_gather_object(all_info, my_info)

        # 在每个rank上，根据收集到的信息构建3D矩阵
        # 初始化一个3D列表，用一个哨兵值（如-1）填充
        matrix = [[[-1 for _ in range(mpu.get_tensor_model_parallel_world_size())] for _ in range(mpu.get_data_parallel_world_size())] for _ in range(mpu.get_pipeline_model_parallel_world_size())]

        # 遍历收集到的信息，填充矩阵
        for info in all_info:
            pp, dp, tp = info['coords']
            global_rank = info['rank']
            local_rank = info['local_rank']
            matrix[pp][dp][tp] = (global_rank, local_rank)

        if dist.get_rank() == 0:
            self.logger.debug('=== Rank Matrix (Global Rank, Local Rank) ===')
            self.logger.debug('Layout: Pipeline Stages (rows) -> Data Parallel Groups (columns) -> Tensor Parallel Ranks (within cell)')
            for pp_idx, dp_plane in enumerate(matrix):
                self.logger.debug(f'--- Pipeline Stage {pp_idx} ---')
                # Header for TP ranks
                header = "DP \\ TP | " + " | ".join([f"  TP={i}  " for i in range(mpu.get_tensor_model_parallel_world_size())])
                self.logger.debug(header)
                self.logger.debug("-" * len(header))
                
                for dp_idx, tp_row in enumerate(dp_plane):
                    row_str = f"  DP={dp_idx}  | "
                    cells = []
                    for global_rank, local_rank in tp_row:
                        cells.append(f"G={global_rank:2d},L={local_rank:1d}")
                    row_str += " | ".join(cells)
                    self.logger.debug(row_str)
                self.logger.debug("")
            
        return matrix
    
    def __gather_rank_to_ip(self):
        """Gather all ranks' IP addresses into a list.

        Returns:
            List[str]: List of IP addresses indexed by global rank.
        """
        self.logger.info("gathering rank to ip list")
        my_ip = dist_utils.get_ip_address()

        # Prepare a list to hold the IP addresses from all ranks
        world_size = dist.get_world_size()
        all_ips = [None] * world_size

        # Gather the IP address string from every rank into the list
        dist.all_gather_object(all_ips, my_ip)

        if dist.get_rank() == 0:
            self.logger.debug('=== Rank to IP Mapping ===')
            for rank, ip in enumerate(all_ips):
                self.logger.debug(f'Rank {rank:3d}: {ip}')

        return all_ips
    
    def __build_student_meta(self) -> dict:
        self.logger.info("building student meta")
        return {
            "student_pp_size": int(mpu.get_pipeline_model_parallel_world_size()),
            "student_tp": int(mpu.get_tensor_model_parallel_world_size()),
            "student_dp": int(mpu.get_data_parallel_world_size()),
            "rank_matrix": self.matrix,
            "rank_to_ips": self.rank_to_ips,
            "student_vocab_size": self.margs.padded_vocab_size
        }
    
    def __build_teacher_meta(self) -> dict:
        self.logger.info("building teacher meta")
        return {
            "teacher_pp_size": int(mpu.get_pipeline_model_parallel_world_size()),
            "teacher_tp": int(mpu.get_tensor_model_parallel_world_size()),
            "teacher_dp": int(mpu.get_data_parallel_world_size()),
            "teacher_vocab_size": self.margs.padded_vocab_size
        }
    
    def __exchange_and_validate_meta(self):
        self.logger.info("exchanging and validating meta")
        student_meta = None if self.is_teacher else self.__build_student_meta()
        teacher_meta = None if not self.is_teacher else self.__build_teacher_meta()

        if self.ar_data_updater is not None:
            if self.is_teacher:
                student_meta = self.ar_data_updater.actor_reference_exchange_meta(teacher_meta)
                self.logger.info(f"received student meta from student")
            else:
                teacher_meta = self.ar_data_updater.actor_reference_exchange_meta(student_meta)
                self.logger.info(f"received teacher meta from teacher")

        self.logger.debug("waiting for rank 0 exchange meta")
        dist.barrier()
        self.logger.info("broadcasting exchanged meta")
        if self.is_teacher:
            buffer = [None] if dist.get_rank() != 0 else [student_meta]
            dist.broadcast_object_list(buffer, src=0)
            student_meta = buffer[0]
        else:
            buffer = [None] if dist.get_rank() != 0 else [teacher_meta]
            dist.broadcast_object_list(buffer, src=0)
            teacher_meta = buffer[0]

        # ===== Validation: only vocab match and tp/dp divisibility =====
        error_message = None
        try:
            self.logger.info("validating exchanged meta")
            def _ensure(cond: bool, msg: str):
                if not cond:
                    self.logger.error(msg)
                    raise RuntimeError(msg)

            s_tp = int(student_meta["student_tp"])       # student tensor-parallel size
            s_dp = int(student_meta["student_dp"])       # student data-parallel size
            s_vocab = int(student_meta["student_vocab_size"])  # student vocab size (padded)
            t_vocab = int(teacher_meta["teacher_vocab_size"])  # teacher vocab size (padded)

            # Vocab must match between teacher and student
            _ensure(t_vocab == s_vocab, f"teacher/student vocab mismatch: teacher={t_vocab}, student={s_vocab}")

            # tp/dp divisibility between teacher and student
            my_tp = int(mpu.get_tensor_model_parallel_world_size())
            my_dp = int(mpu.get_data_parallel_world_size())
            if self.is_teacher:
                t_tp, t_dp = my_tp, my_dp
                _ensure((t_tp % s_tp == 0) or (s_tp % t_tp == 0), f"incompatible tp sizes: teacher_tp={t_tp}, student_tp={s_tp}")
                _ensure((t_dp % s_dp == 0) or (s_dp % t_dp == 0), f"incompatible dp sizes: teacher_dp={t_dp}, student_dp={s_dp}")
            else:
                # Student will rely on teacher to perform divisibility check across worlds.
                pass

        except Exception as e:
            error_message = f"LogitsExpress meta validation failed: {e}"
            self.logger.error(error_message)

        finally:
            # Cross-world status exchange (two rounds so both sides receive peer status)
            local_status = {
                "error": error_message is not None,
                "message": error_message,
            }
            peer_status = {"error": False, "message": None}

            abort = None
            if self.ar_data_updater is not None:
                peer_status = self.ar_data_updater.actor_reference_exchange_meta(local_status)
                # Determine abort flag on rank 0, then broadcast to all local ranks
                abort = local_status["error"] or peer_status["error"]

            # Share abort decision with all local ranks
            buf = [None]
            if dist.get_rank() == 0:
                buf = [abort]
            dist.broadcast_object_list(buf, src=0)
            abort = bool(buf[0])

            if abort:
                # Final barrier so logs flush roughly together
                dist.barrier()
                msg = error_message if error_message is not None else "Cross-world distillation aborted by peer error"
                raise RuntimeError(msg)

            return student_meta, teacher_meta
        
    def __find_your_neighbors(
        self,
        is_teacher: bool,
        teacher_tp: int,
        teacher_dp: int,
        student_tp: int,
        student_dp: int,
        my_tp_rank: int,
        my_dp_rank: int,
        return_group_id: bool = False,
    ):
        """Compute the *equivalence-class* neighbor set for cross-group distillation.

        This returns the FULL symmetric communication group (teacher + student ranks)
        for the coarse bucket (coarse_tp, coarse_dp) that the current rank belongs to.

        Any rank (teacher or student) mapped into the same coarse bucket obtains an
        IDENTICAL returned list (ordering is deterministic) so they can establish
        consistent connection ordering.

        Parameters
        ----------
        is_teacher : bool
            Whether the calling rank belongs to the teacher process group.
        teacher_tp, teacher_dp : int
            Teacher tensor-parallel and data-parallel sizes.
        student_tp, student_dp : int
            Student tensor-parallel and data-parallel sizes.
        my_tp_rank, my_dp_rank : int
            Local (tp, dp) coordinates of THIS rank inside ITS OWN group.
        return_group_id : bool, default False
            If True, also return a tuple (coarse_tp, coarse_dp) identifying the group.

        Returns
        -------
        neighbors : List[Tuple[bool, int, int]]
            List of (is_teacher, tp_rank, dp_rank) for ALL members (teacher first).
        group_id : Tuple[int, int], optional
            Only when return_group_id=True.

        Raises
        ------
        ValueError
            On non-divisible size relationships or out-of-range rank indices.
        """

        self.logger.info("finding neighbors")

        # ----------------- validation helpers -----------------
        def _validate_dim(teacher_size: int, student_size: int, name: str):
            if not (teacher_size % student_size == 0 or student_size % teacher_size == 0):
                raise ValueError(
                    f"{name} sizes not divisible: teacher_{name}={teacher_size}, student_{name}={student_size}"  # noqa: E501
                )

        _validate_dim(teacher_tp, student_tp, "tp")
        _validate_dim(teacher_dp, student_dp, "dp")

        if is_teacher:
            if not (0 <= my_tp_rank < teacher_tp):
                raise ValueError(f"my_tp_rank={my_tp_rank} out of range for teacher_tp={teacher_tp}")
            if not (0 <= my_dp_rank < teacher_dp):
                raise ValueError(f"my_dp_rank={my_dp_rank} out of range for teacher_dp={teacher_dp}")
        else:
            if not (0 <= my_tp_rank < student_tp):
                raise ValueError(f"my_tp_rank={my_tp_rank} out of range for student_tp={student_tp}")
            if not (0 <= my_dp_rank < student_dp):
                raise ValueError(f"my_dp_rank={my_dp_rank} out of range for student_dp={student_dp}")

        # ----------------- mapping to coarse indices -----------------
        def map_to_coarse(role_is_teacher: bool, r: int, teacher_size: int, student_size: int) -> int:
            if teacher_size == student_size:
                return r
            if teacher_size > student_size:  # teacher finer
                if role_is_teacher:
                    bucket = teacher_size // student_size
                    return r // bucket
                else:  # student already coarse
                    return r
            else:  # student finer
                if role_is_teacher:
                    return r
                else:
                    bucket = student_size // teacher_size
                    return r // bucket

        coarse_tp = map_to_coarse(is_teacher, my_tp_rank, teacher_tp, student_tp)
        coarse_dp = map_to_coarse(is_teacher, my_dp_rank, teacher_dp, student_dp)

        # ----------------- enumerate ranks inside a coarse bucket -----------------
        def enumerate_role_indices(role_is_teacher: bool, coarse_idx: int, teacher_size: int, student_size: int):
            if teacher_size == student_size:
                return [coarse_idx]
            if teacher_size > student_size:  # teacher finer
                if role_is_teacher:
                    bucket = teacher_size // student_size
                    start = coarse_idx * bucket
                    end = (coarse_idx + 1) * bucket
                    return list(range(start, end))
                else:
                    return [coarse_idx]
            else:  # student finer
                if role_is_teacher:
                    return [coarse_idx]
                else:
                    bucket = student_size // teacher_size
                    start = coarse_idx * bucket
                    end = (coarse_idx + 1) * bucket
                    return list(range(start, end))

        teacher_tp_set = enumerate_role_indices(True, coarse_tp, teacher_tp, student_tp)
        teacher_dp_set = enumerate_role_indices(True, coarse_dp, teacher_dp, student_dp)
        student_tp_set = enumerate_role_indices(False, coarse_tp, teacher_tp, student_tp)
        student_dp_set = enumerate_role_indices(False, coarse_dp, teacher_dp, student_dp)

        # ----------------- build full symmetric group -----------------
        members = []
        for t_dp in teacher_dp_set:  # dp-major for stable ordering within role
            for t_tp in teacher_tp_set:
                members.append((True, t_tp, t_dp))
        for s_dp in student_dp_set:
            for s_tp in student_tp_set:
                members.append((False, s_tp, s_dp))

        # Deterministic final ordering: teacher first already ensured, then dp, tp within each role.
        # (If a single global sort is desired, we can uncomment the below line.)
        # members.sort(key=lambda x: (-int(x[0]), x[2], x[1]))

        if return_group_id:
            return members, (coarse_tp, coarse_dp)
        return members
    
    def __create_cross_model_parallel_group(
        self,
        neighbors: list[tuple[bool, int, int]],
        student_matrix,
        student_rank_to_ips,
        student_pp_size: int,
        base_port: int | None = None,
        group_name: str | None = None,
        backend: str | None = None,
        timeout_seconds: int = 6000,
    ):
        """创建教师-学生跨模型并行通信组（ProcessGroup）。

        要点：
        - 按照 "student -> dp -> tp" 的优先级对 neighbors 进行整体排序，并以排序后的列表顺序作为 PG 的 rank 分配。
        - 选举 master：排序后列表中的第一个 student（即 dp、tp 最小的 student）。
        - master 的 endpoint 由 student 世界提供的 matrix 和 rank_to_ips 计算：
            init_method = tcp://{ip}:{base_port + local_rank}
        - 通过 init_custom_process_group 创建独立 PG，并返回句柄与我的 pg_rank。

        参数（通过传参消除跨世界发现依赖）：
        - neighbors: List[(is_teacher: bool, tp: int, dp: int)]
        - student_matrix: 3D 列表（student 世界），matrix[pp][dp][tp] = (global_rank, local_rank)
        - student_rank_to_ips: List[str]（student 世界），下标为 student global rank
        - student_pp_size: int，student 的 pipeline stage 数量，用于取最后 stage 的索引
        - base_port: int，可选，默认取环境 CROSS_BASE_PORT 或 29500
        - group_name: str，可选，未提供时基于 neighbors 生成稳定名称
        - backend: str，可选，默认：CUDA 可用则 "nccl"，否则 "gloo"
        - timeout_seconds: int，默认 600

        返回：
        - dict: { pg, rank, world_size, members, group_name }
        """
        self.logger.info("creating cross model parallel group")

        # ---------------- 参数与环境 ----------------
        is_teacher = self.is_teacher

        if base_port is None:
            try:
                base_port = int(os.getenv("CROSS_BASE_PORT", "29500"))
            except ValueError:
                base_port = 29500

        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        # ---------------- 排序与 PG rank 分配 ----------------
        # 目标排序：student 优先（False < True），再按 dp 升序、tp 升序
        sorted_neighbors = sorted(neighbors, key=lambda x: (int(x[0]), x[2], x[1]))

        world_size = len(sorted_neighbors)
        if world_size == 0:
            raise ValueError("neighbors 为空，无法创建跨模型通信组")

        # 找到所有 student 成员
        student_members = [item for item in sorted_neighbors if item[0] is False]
        if not student_members:
            raise ValueError("neighbors 中没有 student 成员，无法选举 master")

        # 选举 master：排序后第一个 student 即为 (dp,tp) 最小的 student
        master_is_teacher, master_tp, master_dp = student_members[0][0], student_members[0][1], student_members[0][2]
        assert master_is_teacher is False, "master 必须来自 student 侧"

        # ---------------- 计算 master endpoint ----------------
        # 使用 student 世界的 matrix 和 rank_to_ips，取最后一个 pipeline stage
        pp_last = student_pp_size - 1
        # 基础校验
        if not (0 <= pp_last < len(student_matrix)):
            raise IndexError(f"student_pp_size={student_pp_size} 与 student_matrix 不匹配")
        if not (0 <= master_dp < len(student_matrix[pp_last])):
            raise IndexError(f"master_dp={master_dp} 超出 student_matrix 维度")
        if not (0 <= master_tp < len(student_matrix[pp_last][master_dp])):
            raise IndexError(f"master_tp={master_tp} 超出 student_matrix 维度")

        s_grank, s_lrank = student_matrix[pp_last][master_dp][master_tp]
        if not (0 <= s_grank < len(student_rank_to_ips)):
            raise IndexError(f"student global_rank={s_grank} 超出 student_rank_to_ips 范围")

        master_ip = student_rank_to_ips[s_grank]
        master_port = base_port + int(s_lrank)

        init_method = f"tcp://{master_ip}:{master_port}"

        # ---------------- 计算我的 pg_rank ----------------

        me = (is_teacher, mpu.get_tensor_model_parallel_rank(), mpu.get_data_parallel_rank())

        try:
            pg_rank = next(i for i, it in enumerate(sorted_neighbors) if it == me)
        except StopIteration:
            raise ValueError(
                f"当前进程 (role={'teacher' if is_teacher else 'student'}, tp={me[1]}, dp={me[2]}) 不在 neighbors 中，"
                "请检查参数与本地并行坐标是否一致"
            )

        # ---------------- group_name 生成 ----------------
        if group_name is None:
            group_name = f"logits_express_pg_sg_{s_grank}"

        # ---------------- 创建 PG ----------------
        timeout = timedelta(minutes=self.margs.distributed_timeout_minutes)
        self.logger.debug("======create_cross_model_parallel_group")
        self.logger.debug(f"{init_method}, {pg_rank}, {world_size}, {group_name}")
        pg = init_custom_process_group(
            backend=backend,
            init_method=init_method,
            timeout=timeout,
            world_size=world_size,
            rank=pg_rank,
            group_name=group_name,
        )
        self.logger.debug("==cross pg barrier")
        dist.barrier(group=pg)
        self.logger.debug("==cross pg barrier passed")

        return {
            "pg": pg,
            "rank": pg_rank,
            "world_size": world_size,
            "members": sorted_neighbors,
            "group_name": group_name,
        }
    
    def __find_neighbors_and_create_cross_pg(self):
        if mpu.is_pipeline_last_stage():
            s_meta = self.student_meta
            t_meta = self.teacher_meta

            S_tp = int(s_meta["student_tp"])
            S_dp = int(s_meta["student_dp"])

            T_tp = int(t_meta["teacher_tp"])
            T_dp = int(t_meta["teacher_dp"])

            neighbors, _gid = self.__find_your_neighbors(
                is_teacher=self.is_teacher,
                teacher_tp=T_tp,
                teacher_dp=T_dp,
                student_tp=S_tp,
                student_dp=S_dp,
                my_tp_rank=mpu.get_tensor_model_parallel_rank(),
                my_dp_rank=mpu.get_data_parallel_rank(),
                return_group_id=True,
            )
            self.logger.info(f"[Role={'teacher' if self.is_teacher else 'student'} GRank={dist.get_rank()}] GroupID={_gid} Neighbors={neighbors}")
        self.logger.debug("waiting for pp last stage to find neighbors")
        dist.barrier()

        logits_express_pg_info = None
        if mpu.is_pipeline_last_stage():
            logits_express_pg_info = self.__create_cross_model_parallel_group(
                neighbors,
                student_matrix=self.student_meta["rank_matrix"],
                student_rank_to_ips=self.student_meta["rank_to_ips"],
                student_pp_size=int(self.student_meta["student_pp_size"]),
                base_port=self.margs.logits_pg_base_port,
                backend='nccl'
            )
        self.logger.debug("waiting for pp last stage to create cross pg")
        dist.barrier()
        return logits_express_pg_info
    
    def __tensor_to_tensor(
        self,
        dst: torch.Tensor, src: torch.Tensor,
        a: int, b: int, c: int, d: int,
        i: int, j: int, k: int, l: int
    ):
        """将 src[i:j, :, k:l] 复制到 dst[a:b, :, c:d]。

        仅做形状与边界的轻校验，核心是一次连续块 copy_，便于二次封装成任意 Kd×Kt 的组合拷贝。
        """
        # 边界校验（可按需放宽以追求极致性能）
        assert 0 <= a <= b <= dst.shape[0], f"dst batch slice [{a}:{b}] 超界"
        assert 0 <= c <= d <= dst.shape[2], f"dst vocab slice [{c}:{d}] 超界"
        assert 0 <= i <= j <= src.shape[0], f"src batch slice [{i}:{j}] 超界"
        assert 0 <= k <= l <= src.shape[2], f"src vocab slice [{k}:{l}] 超界"

        # 中间维长度必须一致（seq 长度一致）
        assert dst.shape[1] == src.shape[1], "dst 与 src 的 seq 维必须一致"

        # 目标/源子块形状一致
        assert (b - a) == (j - i), "batch 子段长度不一致"
        assert (d - c) == (l - k), "vocab 子段长度不一致"

        # 实际拷贝
        dst[a:b, :, c:d].copy_(src[i:j, :, k:l])
    
    def teacher_send_student_receive(self, logits = None):
        """多轮 all_to_all_single 的 Teacher→Student 传输（零复制版本）。

        思路与约定：
        - 仍使用 create_cross_model_parallel_group 固定的成员顺序（students 在前，dp→tp 升序），并通过 _CROSS_PG_MEMBERS 取回。
        - 与 teacher_send_student_receive 的差异：按学生逐个轮次进行通信。
        每一轮仅面向单个 Student：所有 Teacher 向该 Student 发送自己的 logits，其他 Students 在该轮 recv_splits 全为 0。
        这样 Teacher 不需要一次性复制拼接到大 buffer（零复制），仅重复参与多轮通信以覆盖全部 Students。
        - 返回语义：Teacher 返回 None；Student 返回按教师顺序的张量列表（只需在“属于自己的一轮”解析即可）。
        - 仍做一次轻量的 all_gather_object 交换/校验元信息（numel、dtype、shape），便于 Student 分配接收缓冲。
        - 与原版不同之处：学生在属于自己的轮次里，将收到的 teacher 列表重组为自身分片形状 (B_s, L, V_s_shard) 并在函数末尾返回；其他轮次继续参与通信但不返回数据。
        """

        # teacher logits的形状为(transfer_bs//dp_size_t, seq_len, vocab_size//tp_size_t)
        # student在函数最后得到的logits的形状应该为(transfer_bs//dp_size_s, seq_len, vocab_size//tp_size_s)
        # 保证transfer_bs同时被dp_size_t和dp_size_s整除
        # 执行该函数时，teacher和student的都已知道自己和对方的dp_size和tp_size

        pg = self.logits_express_pg_info["pg"]
        pg_rank = self.logits_express_pg_info["rank"]

        def student_recover_logits_from_teacher(teacher_logits_list: list[torch.Tensor]):
            """用 Kd×Kt 个切片复制将 teacher 列表重组为当前学生分片。

            列表顺序要求：dp 主序、tp 次序（与 members 中 teacher 的 dp→tp 排序一致）。
            """
            # 1) 解析集合与当前学生在组内的局部索引
            teacher_tps = sorted({tp for (is_t, tp, dp) in members if is_t})
            teacher_dps = sorted({dp for (is_t, tp, dp) in members if is_t})
            student_tps = sorted({tp for (is_t, tp, dp) in members if not is_t})
            student_dps = sorted({dp for (is_t, tp, dp) in members if not is_t})

            T_tp = len(teacher_tps)
            T_dp = len(teacher_dps)
            S_tp = len(student_tps)
            S_dp = len(student_dps)

            my_s_tp = my_member[1]
            my_s_dp = my_member[2]
            try:
                kt = student_tps.index(my_s_tp)
                kd = student_dps.index(my_s_dp)
            except ValueError:
                raise RuntimeError("当前学生 tp/dp 不在该等价组的 student_tps/student_dps 内")

            if len(teacher_logits_list) != T_dp * T_tp:
                raise ValueError("teacher_logits_list 长度必须为 T_dp*T_tp，并以 dp 主序、tp 次序排列")

            # 2) 基本形状与目标形状
            B_t, L, V_t = base_shape

            if T_dp > S_dp:
                Kd = T_dp // S_dp
                B_s = Kd * B_t
            elif T_dp == S_dp:
                Kd = 1
                B_s = B_t
            else:
                Kd = S_dp // T_dp
                assert B_t % Kd == 0, "B_t 必须能被 Kd 整除"
                B_s = B_t // Kd

            if T_tp > S_tp:
                Kt = T_tp // S_tp
                V_s = Kt * V_t
            elif T_tp == S_tp:
                Kt = 1
                V_s = V_t
            else:
                Kt = S_tp // T_tp
                assert V_t % Kt == 0, "V_t 必须能被 Kt 整除"
                V_s = V_t // Kt

            out = torch.empty((B_s, L, V_s), dtype=base_dtype, device=default_device)

            # 3) 双重小循环：dp 段×tp 段
            # dp 轴参数（决定 a:b 与 i:j 以及 j_local）
            def dp_params(m: int):
                if T_dp > S_dp:
                    # 拼 batch，第 m 段
                    j_local = kd * Kd + m
                    a = m * B_t
                    b = a + B_t
                    i = 0
                    j = B_t
                elif T_dp == S_dp:
                    j_local = kd
                    a, b = 0, B_t
                    i, j = 0, B_t
                else:
                    # 切 batch
                    j_local = kd // Kd
                    chunk = B_t // Kd
                    a, b = 0, chunk
                    i = (kd % Kd) * chunk
                    j = i + chunk
                return j_local, a, b, i, j

            # tp 轴参数（决定 c:d 与 k:l 以及 i_local）
            def tp_params(n: int):
                if T_tp > S_tp:
                    # 拼 vocab，第 n 段
                    i_local = kt * Kt + n
                    c = n * V_t
                    d = c + V_t
                    k, l = 0, V_t
                elif T_tp == S_tp:
                    i_local = kt
                    c, d = 0, V_t
                    k, l = 0, V_t
                else:
                    # 切 vocab
                    i_local = kt // Kt
                    width = V_t // Kt
                    c, d = 0, width
                    k = (kt % Kt) * width
                    l = k + width
                return i_local, c, d, k, l

            for m in range(Kd):
                j_local, a, b, i_beg, i_end = dp_params(m)
                for n in range(Kt):
                    i_local, c, d, k_beg, k_end = tp_params(n)
                    idx = j_local * T_tp + i_local
                    src = teacher_logits_list[idx]
                    self.__tensor_to_tensor(out, src, a, b, c, d, i_beg, i_end, k_beg, k_end)

            return out

        # ---------------- 实际多轮通信与恢复 ----------------
        if pg is None:
            raise ValueError("pg 不能为空")

        members = self.logits_express_pg_info["members"]
        world_size = len(members)

        my_rank = pg_rank
        my_member = members[my_rank]
        my_is_teacher = bool(my_member[0])
        teacher_indices = [i for i, m in enumerate(members) if m[0] is True]
        student_indices = [i for i, m in enumerate(members) if m[0] is False]

        if not teacher_indices:
            raise RuntimeError("通信组内没有 Teacher 成员")
        if not student_indices:
            raise RuntimeError("通信组内没有 Student 成员")

        default_device = dist_utils.get_device()

        # 元信息交换，拿到 base 形状/类型
        if my_is_teacher:
            if logits is None or not isinstance(logits, torch.Tensor):
                raise ValueError("Teacher 侧必须提供 logits 张量")
            if logits.device.type != default_device.type:
                logits = logits.to(default_device)
            t_numel = int(logits.numel())
            t_dtype = str(logits.dtype)
            t_shape = tuple(logits.shape)
        else:
            t_numel = 0
            t_dtype = None
            t_shape = None

        local_meta = {"is_teacher": my_is_teacher, "numel": t_numel, "dtype": t_dtype, "shape": t_shape}
        meta_list = [None] * world_size
        dist.all_gather_object(meta_list, local_meta, group=pg)

        teacher_metas = [meta_list[i] for i in teacher_indices]
        base_numel = teacher_metas[0]["numel"]
        base_dtype_str = teacher_metas[0]["dtype"]
        base_shape = teacher_metas[0]["shape"]

        def _dtype_from_str(s: str):
            if s is None:
                return None
            name = s.split(".")[-1]
            if not hasattr(torch, name):
                raise ValueError(f"无法解析 dtype: {s}")
            return getattr(torch, name)

        base_dtype = _dtype_from_str(base_dtype_str)

        for tm in teacher_metas:
            if tm["numel"] != base_numel:
                raise ValueError("所有 Teacher 的 logits 元素数必须一致")
            if tm["dtype"] != base_dtype_str:
                raise ValueError("所有 Teacher 的 dtype 必须一致")
            if tm["shape"] != base_shape:
                raise ValueError("所有 Teacher 的 logits 形状必须一致")

        # Teacher 侧：准备展平只读视图
        if my_is_teacher:
            flat = logits.contiguous().view(-1)

        # 学生结果（仅在属于自己的轮次填充）
        student_result = None

        # 按 student 顺序逐轮通信
        self.logger.debug(f"base_numel is : {base_numel}")
        for s_idx in student_indices:
            if my_is_teacher:
                send_splits = [base_numel if i == s_idx else 0 for i in range(world_size)]
                recv_splits = [0] * world_size
                send_buffer = flat
                recv_buffer = torch.empty(0, dtype=base_dtype, device=default_device)
            else:
                send_splits = [0] * world_size
                if my_rank == s_idx:
                    recv_splits = [base_numel if i in teacher_indices else 0 for i in range(world_size)]
                    total_recv = sum(recv_splits)
                    recv_buffer = torch.empty(total_recv, dtype=base_dtype, device=default_device)
                else:
                    recv_splits = [0] * world_size
                    recv_buffer = torch.empty(0, dtype=base_dtype, device=default_device)
                send_buffer = torch.empty(0, dtype=base_dtype, device=default_device)

            print("===teacher_send_student_receive_2 before all to all")
            print(send_splits)
            print(recv_splits)
            dist.all_to_all_single(
                output=recv_buffer,
                input=send_buffer,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=pg,
            )

            # 仅在属于自己的轮次：解析 teacher 列表并重构为学生分片
            if (not my_is_teacher) and (my_rank == s_idx):
                # 拆分为 teacher 顺序的视图列表
                t_list = []
                offset = 0
                B_t, L, V_t_shard = base_shape
                for i in teacher_indices:
                    cnt = recv_splits[i]
                    if cnt > 0:
                        chunk = recv_buffer.narrow(0, offset, cnt).view(B_t, L, V_t_shard)
                        t_list.append(chunk)
                        offset += cnt
                # 调用重构函数
                student_result = student_recover_logits_from_teacher(t_list)

        return student_result if not my_is_teacher else None

    def _student_recover_topk_from_teacher(
        self,
        teacher_indices_list: list[torch.Tensor],
        teacher_values_list: list[torch.Tensor],
        members: list[tuple[bool, int, int]],
        my_member: tuple[bool, int, int],
    ):
        teacher_tps = sorted({tp for (is_t, tp, _dp) in members if is_t})
        teacher_dps = sorted({dp for (is_t, _tp, dp) in members if is_t})
        student_dps = sorted({dp for (is_t, _tp, dp) in members if not is_t})

        T_tp = len(teacher_tps)
        T_dp = len(teacher_dps)
        S_dp = len(student_dps)

        if len(teacher_indices_list) != T_dp * T_tp or len(teacher_values_list) != T_dp * T_tp:
            raise ValueError("teacher topk list 长度应为 T_dp*T_tp")

        B_t, L, Kp = teacher_indices_list[0].shape
        my_s_dp = my_member[2]
        kd = student_dps.index(my_s_dp)

        dp_ranges = []
        if T_dp > S_dp:
            Kd = T_dp // S_dp
            B_s = Kd * B_t
            for m in range(Kd):
                j_local = kd * Kd + m
                a = m * B_t
                b = a + B_t
                dp_ranges.append((j_local, 0, B_t, a, b))
        elif T_dp == S_dp:
            B_s = B_t
            dp_ranges.append((kd, 0, B_t, 0, B_t))
        else:
            Kd = S_dp // T_dp
            if B_t % Kd != 0:
                raise ValueError("B_t 必须能被 Kd 整除")
            chunk = B_t // Kd
            j_local = kd // Kd
            src_start = (kd % Kd) * chunk
            src_end = src_start + chunk
            B_s = chunk
            dp_ranges.append((j_local, src_start, src_end, 0, chunk))

        # 新语义：teacher 侧已给出“全局 topK + label 槽”的固定形状 K'（K+1），
        # recover 仅做 DP 对齐；TP 维不再做拆分、拼接、过滤或再 topK。
        out_idx = torch.empty((B_s, L, int(Kp)), dtype=torch.long, device=teacher_indices_list[0].device)
        out_val = torch.empty((B_s, L, int(Kp)), dtype=teacher_values_list[0].dtype, device=teacher_values_list[0].device)

        # teacher 各 TP rank 结果应一致（均为全局结果），recover 时固定选择第一个 teacher TP 即可。
        ref_i_local = 0

        for j_local, src_start, src_end, dst_start, dst_end in dp_ranges:
            flat_idx = j_local * T_tp + ref_i_local
            out_idx[dst_start:dst_end] = teacher_indices_list[flat_idx][src_start:src_end]
            out_val[dst_start:dst_end] = teacher_values_list[flat_idx][src_start:src_end]

        return out_idx, out_val

    def teacher_send_student_receive_topk(self, indices_global=None, values=None):
        pg = self.logits_express_pg_info["pg"]
        pg_rank = self.logits_express_pg_info["rank"]
        members = self.logits_express_pg_info["members"]
        world_size = len(members)

        my_rank = pg_rank
        my_member = members[my_rank]
        my_is_teacher = bool(my_member[0])
        teacher_indices = [i for i, m in enumerate(members) if m[0] is True]
        student_indices = [i for i, m in enumerate(members) if m[0] is False]

        if not teacher_indices or not student_indices:
            raise RuntimeError("通信组 teacher/student 成员异常")

        default_device = dist_utils.get_device()
        if my_is_teacher:
            if indices_global is None or values is None:
                raise ValueError("Teacher 侧必须提供 indices_global 和 values")
            if indices_global.device.type != default_device.type:
                indices_global = indices_global.to(default_device)
            if values.device.type != default_device.type:
                values = values.to(default_device)
            idx_shape = tuple(indices_global.shape)
            val_shape = tuple(values.shape)
            local_meta = {
                "is_teacher": True,
                "idx_numel": int(indices_global.numel()),
                "idx_dtype": str(indices_global.dtype),
                "idx_shape": idx_shape,
                "val_numel": int(values.numel()),
                "val_dtype": str(values.dtype),
                "val_shape": val_shape,
            }
        else:
            local_meta = {
                "is_teacher": False,
                "idx_numel": 0,
                "idx_dtype": None,
                "idx_shape": None,
                "val_numel": 0,
                "val_dtype": None,
                "val_shape": None,
            }

        meta_list = [None] * world_size
        dist.all_gather_object(meta_list, local_meta, group=pg)
        teacher_metas = [meta_list[i] for i in teacher_indices]

        # 显式 schema 校验，避免在模式不一致（例如 teacher 走 dense API, student 走 topk API）时触发 KeyError。
        required_topk_meta_keys = {
            "is_teacher",
            "idx_numel",
            "idx_dtype",
            "idx_shape",
            "val_numel",
            "val_dtype",
            "val_shape",
        }
        for idx, tm in zip(teacher_indices, teacher_metas):
            if not isinstance(tm, dict):
                raise RuntimeError(
                    f"Invalid teacher meta at pg_rank={idx}: type={type(tm)}. "
                    "Likely actor/reference are not in the same logits transfer mode."
                )
            missing = [k for k in required_topk_meta_keys if k not in tm]
            if missing:
                raise RuntimeError(
                    f"Invalid topk meta schema from teacher pg_rank={idx}, missing keys={missing}, meta_keys={list(tm.keys())}. "
                    "This usually means teacher and student are calling different communication APIs "
                    "(dense vs topk). Please ensure both sides use the same gkd_sparse_topk_enabled setting."
                )

        base_idx_numel = teacher_metas[0]["idx_numel"]
        base_val_numel = teacher_metas[0]["val_numel"]
        base_idx_dtype_str = teacher_metas[0]["idx_dtype"]
        base_val_dtype_str = teacher_metas[0]["val_dtype"]
        base_idx_shape = teacher_metas[0]["idx_shape"]
        base_val_shape = teacher_metas[0]["val_shape"]

        for tm in teacher_metas:
            if tm["idx_numel"] != base_idx_numel or tm["val_numel"] != base_val_numel:
                raise ValueError("Teacher topk numel 不一致")
            if tm["idx_dtype"] != base_idx_dtype_str or tm["val_dtype"] != base_val_dtype_str:
                raise ValueError("Teacher topk dtype 不一致")
            if tm["idx_shape"] != base_idx_shape or tm["val_shape"] != base_val_shape:
                raise ValueError("Teacher topk shape 不一致")

        def _dtype_from_str(s: str):
            name = s.split(".")[-1]
            if not hasattr(torch, name):
                raise ValueError(f"无法解析 dtype: {s}")
            return getattr(torch, name)

        idx_dtype = _dtype_from_str(base_idx_dtype_str)
        val_dtype = _dtype_from_str(base_val_dtype_str)

        if my_is_teacher:
            flat_idx = indices_global.contiguous().view(-1)
            flat_val = values.contiguous().view(-1)

        student_indices_result = None
        student_values_result = None

        for s_idx in student_indices:
            if my_is_teacher:
                send_splits_idx = [base_idx_numel if i == s_idx else 0 for i in range(world_size)]
                send_splits_val = [base_val_numel if i == s_idx else 0 for i in range(world_size)]
                recv_splits_idx = [0] * world_size
                recv_splits_val = [0] * world_size
                send_idx = flat_idx
                send_val = flat_val
                recv_idx = torch.empty(0, dtype=idx_dtype, device=default_device)
                recv_val = torch.empty(0, dtype=val_dtype, device=default_device)
            else:
                send_splits_idx = [0] * world_size
                send_splits_val = [0] * world_size
                if my_rank == s_idx:
                    recv_splits_idx = [base_idx_numel if i in teacher_indices else 0 for i in range(world_size)]
                    recv_splits_val = [base_val_numel if i in teacher_indices else 0 for i in range(world_size)]
                    recv_idx = torch.empty(sum(recv_splits_idx), dtype=idx_dtype, device=default_device)
                    recv_val = torch.empty(sum(recv_splits_val), dtype=val_dtype, device=default_device)
                else:
                    recv_splits_idx = [0] * world_size
                    recv_splits_val = [0] * world_size
                    recv_idx = torch.empty(0, dtype=idx_dtype, device=default_device)
                    recv_val = torch.empty(0, dtype=val_dtype, device=default_device)
                send_idx = torch.empty(0, dtype=idx_dtype, device=default_device)
                send_val = torch.empty(0, dtype=val_dtype, device=default_device)

            dist.all_to_all_single(
                output=recv_idx,
                input=send_idx,
                output_split_sizes=recv_splits_idx,
                input_split_sizes=send_splits_idx,
                group=pg,
            )
            dist.all_to_all_single(
                output=recv_val,
                input=send_val,
                output_split_sizes=recv_splits_val,
                input_split_sizes=send_splits_val,
                group=pg,
            )

            if (not my_is_teacher) and (my_rank == s_idx):
                t_idx_list = []
                t_val_list = []
                offset_idx = 0
                offset_val = 0
                B_t, L, Kp = base_idx_shape
                for i in teacher_indices:
                    cnt_idx = recv_splits_idx[i]
                    cnt_val = recv_splits_val[i]
                    if cnt_idx > 0:
                        t_idx_list.append(recv_idx.narrow(0, offset_idx, cnt_idx).view(B_t, L, Kp))
                        offset_idx += cnt_idx
                    if cnt_val > 0:
                        t_val_list.append(recv_val.narrow(0, offset_val, cnt_val).view(B_t, L, Kp))
                        offset_val += cnt_val

                student_indices_result, student_values_result = self._student_recover_topk_from_teacher(
                    teacher_indices_list=t_idx_list,
                    teacher_values_list=t_val_list,
                    members=members,
                    my_member=my_member,
                )

        if my_is_teacher:
            return None
        return student_indices_result, student_values_result

    def simulate_student_segments_via_neighbors(self, train_bs, iter_num: int, neighbors: list[tuple[bool, int, int]] | None = None) -> list[tuple[int, int]]:
        """基于“实际通信邻居列表 + 轮次 iter_num”的段模拟（按通信顺序精确对齐）。

        设计要点：
        - 完全复用 cross PG 的成员顺序（teacher 在前，dp→tp），只在“教师成员”上按 dp 去重保序，得到本 coarse 组的 teacher-dp 序列；
        - 用 iter_num 确定本轮全局基准偏移 base = iter_num * transfer_bs；
        - 逐个 teacher-dp 生成 (start, length)；Student 更细时从该 teacher-dp 的 B_t 中切 1/Kd 子段，子段序号由“我在组内的局部学生下标”决定。

        输入
        ----
        transfer_bs : int
            本轮 Teacher→Student 汇总发送的总 batch 大小（按 teacher 全局 DP 等分）。
        iter_num : int
            本轮序号（0-based），与教师端 compute_and_send 中的 i 对齐。
        neighbors : Optional[List[(is_teacher, tp, dp)]]
            传入则使用该列表；否则默认取 self.logits_express_pg_info["members"]。

        返回
        ----
        List[(start, length)] 按通信顺序排列的段；start 为“全局行号”，= base + t_dp*B_t (+ 子块偏移)。
        """

        transfer_bs = self.margs.logits_transfer_batch_size
        self.logger.debug(f"train_bs: {train_bs}, transfer_bs: {transfer_bs}")

        # 1) 读取成员顺序（必须已创建 cross PG）
        if neighbors is None:
            if self.logits_express_pg_info is None or "members" not in self.logits_express_pg_info:
                raise RuntimeError("cross PG 尚未建立，无法基于 neighbors 模拟段")
            neighbors = self.logits_express_pg_info["members"]

        # 2) 全局 DP 常量与本轮偏移
        T_dp_total = int(self.teacher_meta["teacher_dp"]) if self.teacher_meta is not None else int(mpu.get_data_parallel_world_size())
        S_dp_total = int(self.student_meta["student_dp"]) if self.student_meta is not None else int(mpu.get_data_parallel_world_size())

        if transfer_bs % T_dp_total != 0:
            raise ValueError(f"transfer_bs={transfer_bs} 必须能被 teacher_dp_total={T_dp_total} 整除")
        B_t = transfer_bs // T_dp_total

        # 3) 提取组内 teacher-dp 的“去重保序”列表，以及 student-dp 列表
        teacher_dps_in_group: list[int] = []
        seen = set()
        for is_t, _tp, dp in neighbors:
            if is_t and dp not in seen:
                teacher_dps_in_group.append(dp)
                seen.add(dp)

        student_dps_in_group: list[int] = []
        seen_s = set()
        for is_t, _tp, dp in neighbors:
            if (not is_t) and dp not in seen_s:
                student_dps_in_group.append(dp)
                seen_s.add(dp)

        my_dp = int(mpu.get_data_parallel_rank())
        try:
            kd_local = student_dps_in_group.index(my_dp)
        except ValueError:
            raise RuntimeError(f"当前 student dp={my_dp} 不在本 coarse 组 student 成员中：{student_dps_in_group}")

        segments: list[tuple[int, int]] = []

        # 4) 三种关系分别处理（严格按 teacher_dps_in_group 顺序）
        if T_dp_total > S_dp_total:
            # 老师更细：拼接组内所有 teacher-dp 的整块
            for t_dp in teacher_dps_in_group:
                # 先判断该t_dp对于train batch的offset
                offset = train_bs // T_dp_total * t_dp
                # 再判断该t_dp在当前iter_num下的offset
                local_offset = B_t * iter_num
                start = offset + local_offset
                segments.append((start, B_t))

        elif T_dp_total == S_dp_total:
            # 一对一：仅一个 teacher-dp
            if not teacher_dps_in_group:
                raise RuntimeError("邻居列表中未发现 teacher 成员")
            assert len(teacher_dps_in_group) == 1, "T_dp_total == S_dp_total 时，teacher_dps_in_group 应仅含单一成员"
            t_dp = teacher_dps_in_group[0]
            # 先判断该t_dp对于train batch的offset
            offset = train_bs // T_dp_total * t_dp
            # 再判断该t_dp在当前iter_num下的offset
            local_offset = B_t * iter_num
            start = offset + local_offset
            segments.append((start, B_t))

        else:
            # 学生更细：从唯一 teacher-dp 的 B_t 中切 1/Kd 子段
            Kd = S_dp_total // T_dp_total
            if B_t % Kd != 0:
                raise ValueError(f"B_t={B_t} 必须能被 Kd={Kd} 整除 (S_dp_total/T_dp_total)")
            chunk = B_t // Kd
            if not teacher_dps_in_group:
                raise RuntimeError("邻居列表中未发现 teacher 成员")
            t_dp = teacher_dps_in_group[0]
            assert len(teacher_dps_in_group) == 1, "T_dp_total < S_dp_total 时，teacher_dps_in_group 应仅含单一成员"
            # 先判断该t_dp对于train batch的offset
            offset = train_bs // T_dp_total * t_dp
            # 再判断该t_dp在当前iter_num下的offset
            local_offset = B_t * iter_num
            chunk_offset = kd_local * chunk
            start = offset + local_offset + chunk_offset
            segments.append((start, chunk))

        return segments