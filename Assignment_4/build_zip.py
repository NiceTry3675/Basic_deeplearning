"""
제출용 코드 압축파일 생성: 202258096_임준현_딥러닝응용_과제2코드.zip
  python build_zip.py
포함: code/(py, sh, README, requirements.txt), data/mnist.npz, final_models/(트랙 A·B), jobs/queue.txt,
      results/runs, results/logs, results/report, results/figures, results/teachers의 OOF logit 2개
제외: __pycache__, 쓰지 않는 runq.sh, 중간 체크포인트(results/ckpt), fold별 teacher logit, 분석용 val logit
"""
import glob
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "202258096_임준현_딥러닝응용_과제2코드.zip")

files = []
files += sorted(f for f in glob.glob(os.path.join(HERE, "code", "*")) if os.path.isfile(f)
                and os.path.basename(f) != "runq.sh" and f.endswith((".py", ".sh", ".md", ".txt")))
files += [os.path.join(HERE, "data", "mnist.npz")]
files += sorted(glob.glob(os.path.join(HERE, "final_models", "*.pt")))
files += [os.path.join(HERE, "jobs", "queue.txt")]
for sub in ["runs", "logs", "report", "figures"]:
    files += sorted(f for f in glob.glob(os.path.join(HERE, "results", sub, "*")) if os.path.isfile(f))
files += [os.path.join(HERE, "results", "teachers", n) for n in ["t_res_full_oof5.pt", "t_res_dev_oof5.pt"]]

missing = [f for f in files if not os.path.exists(f)]
assert not missing, missing
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, os.path.relpath(f, HERE))
size = os.path.getsize(OUT) / 1e6
print(f"saved {OUT} ({len(files)} files, {size:.1f} MB)")
groups = {}
for f in files:
    k = os.path.relpath(f, HERE).split(os.sep)[0]
    if k == "results":
        k = "/".join(os.path.relpath(f, HERE).split(os.sep)[:2])
    groups[k] = groups.get(k, 0) + 1
print(groups)
