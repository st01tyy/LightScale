import requests
import json
import re
import difflib


def verify_bash(url, question, response, ground_truth) -> bool:
    ground_truth_dict = json.loads(ground_truth)

    action_input_token = "Action Input:"
    if "Action Input:" in response:
        response = response.split(action_input_token)[-1]
    pattern = r"```bash(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        print("Bash command not found! question id is: ", ground_truth_dict["id"])
        return False
    
    code = matches[0].strip()
    if ground_truth_dict["cmd"].strip() == code:
        print("Bash command is equal! question id is: ", ground_truth_dict["id"])
        return True
    
    run_code = f"{ground_truth_dict['init_shell']}\n{code}"
    response = requests.post(
        f"http://{url}/run_code",
        json={
            "code": run_code,
            "language": "bash",
        },
    )
    res = response.json()
    if (
        res["status"] != "Success"
        or res["run_result"]["return_code"] != 0
        or not res["run_result"]["stdout"]
    ):
        print("Bash command executed failed! question id is: ", ground_truth_dict["id"]) 
        return False

    cur_res: str = res["run_result"]["stdout"].strip()
    exc_res: str = ground_truth_dict["stdout"].strip()
    sim_radio: float = difflib.SequenceMatcher(None, cur_res, exc_res).ratio()
    if "/tmp" in exc_res:
        print(f"Bash command find /tmp! sim radio is: {sim_radio} and question id is: {ground_truth_dict['id']}")  
        return sim_radio >= 0.55
    print(f"Bash command sim radio is: {sim_radio} and question id is: {ground_truth_dict['id']}")
    return sim_radio >= 0.75


if __name__ == '__main__':
    verify_bash("localhost:9090", "", 
                "For example, a result like `0022` means the default permissions for new files are `644` (since `666 - 022 = 644` for files, and `777 - 022 = 755` for directories).\n\n```bash\numask\n```\nAction: code_interpreter\nAction Input: ```bash\numask\n```", 
                "{\"cmd\": \"umask\", \"init_shell\": \"\", \"stdout\": \"0022\\n\", \"id\": \"cm-bash-8432\"}")
