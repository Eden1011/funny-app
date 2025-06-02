import psutil
import time
from collections import defaultdict

"""
Module consisting of various ways to enhance player happiness
"""

def terminate_client():
    target_processes = [
        "LeagueClient.exe",
        "LeagueClientUx.exe",
        "LeagueClientUxRender.exe"
    ]

    results = {
        "terminated": [],
        "not_found": [],
        "access_denied": [],
        "errors": []
    }

    found_processes = []

    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                process_name = proc.info['name']

                if process_name in target_processes:
                    pid = proc.info['pid']
                    found_processes.append({
                        'process': proc,
                        'name': process_name,
                        'pid': pid
                    })


            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

    except Exception as e:

        results["errors"].append({
            "name": "process_scan",
            "error": str(e)
        })

    if not found_processes:

        for target in target_processes:
            results["not_found"].append(target)
        return results

    for proc_info in found_processes:
        proc = proc_info['process']
        process_name = proc_info['name']
        pid = proc_info['pid']

        try:
            if not proc.is_running():
                results["terminated"].append({
                    "name": process_name,
                    "pid": pid,
                    "status": "already_gone"
                })
                continue

            proc.terminate()

            proc.wait(timeout=3)

            results["terminated"].append({
                "name": process_name,
                "pid": pid,
                "status": "terminated"
            })

        except psutil.TimeoutExpired:
            try:
                proc.kill()

                results["terminated"].append({
                    "name": process_name,
                    "pid": pid,
                    "status": "force_killed"
                })
            except Exception as e:

                results["errors"].append({
                    "name": process_name,
                    "pid": pid,
                    "error": str(e)
                })

        except psutil.AccessDenied:
            results["access_denied"].append({
                "name": process_name,
                "pid": pid
            })

        except psutil.NoSuchProcess:
            results["terminated"].append({
                "name": process_name,
                "pid": pid,
                "status": "already_gone"
            })

        except Exception as e:
            results["errors"].append({
                "name": process_name,
                "pid": pid,
                "error": str(e)
            })

    remaining_processes = []

    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                process_name = proc.info['name']
                if process_name in target_processes:
                    remaining_processes.append({
                        "name": process_name,
                        "pid": proc.info['pid']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception:
        pass

    found_names = set(item["name"] for item in found_processes)
    for target in target_processes:
        if target not in found_names:
            results["not_found"].append(target)

    if results["terminated"]:

        by_name = defaultdict(list)
        for item in results["terminated"]:
            by_name[item['name']].append(item)

    return results


def is_client_running():
    target_processes = [
        "LeagueClient.exe",
        "LeagueClientUx.exe",
        "LeagueClientUxRender.exe"
    ]

    running_processes = []

    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                process_name = proc.info['name']
                if process_name in target_processes:
                    running_processes.append({
                        "name": process_name,
                        "pid": proc.info['pid']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        print(f"Error checking processes: {e}")

    return {
        "is_running": len(running_processes) > 0,
        "processes": running_processes,
        "count": len(running_processes)
    }


def chill():
    status = is_client_running()

    if status["is_running"]:
        results = terminate_client()

        if results["terminated"]:
            print("Time to take a break and smile!")
        else:
            print("Couldn't terminate League client.")
    else:
        print("Either League isn't running, or you're already free!")


if __name__ == "__main__":
    chill()