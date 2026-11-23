"""Hand off this already registered desktop study to the standard local runner.

The workbench performs the server requests in its signed-in session. This process
consumes the returned non-credential documents and never handles an account token.
"""
import argparse
from pathlib import Path
import re
import shutil
import time

from clt_role_reread_materials import ENTRY, HERE, OUT, SIZES, read, sha
from run_routine_test import run, ROOT
from deploy.storage_policy import check_write, checked_dir, storage_dir
from deploy.test_archive_client import save
from shared.dist_protocol.test_archive import digest, validate_plan


def verify_receipt(receipt, submission):
    fields = {k:v for k,v in receipt.items() if k not in ("receipt_sha256","readback_verified","cleanup_allowed")}
    fields["cleanup_allowed"] = not receipt.get("missing_core_roles")
    if (receipt.get("run_id")!=submission["run_id"] or not receipt.get("readback_verified")
            or receipt.get("manifest_sha256")!=submission["manifest_sha256"] or digest(fields)!=receipt.get("receipt_sha256")):
        raise ValueError("server receipt, local manifest or readback mismatch")
    return receipt


class DesktopDocuments:
    def __init__(self, stage, bundle, output=OUT):
        self.stage = stage
        self.bundle = bundle
        self.output = output

    def start(self, preflight, run_id, decision, reason, source):
        registration = self.bundle["run"]
        if (preflight!=self.bundle["preflight"] or registration["id"]!=run_id or registration["decision"]!=decision
                or registration["reason"]!=reason or registration["source_run_id"]!=source):
            raise ValueError("desktop registration does not correspond to this execution")
        return registration

    def submit(self, run_id, manifest_path, status, summary):
        package = checked_dir(self.output/f"{self.stage}-package", 512*2**20, operation="已登记实验的归档包")
        files = []
        for entry in read(manifest_path)["files"]:
            name = entry["path"]
            if entry["retention"]=="scratch":
                continue
            if Path(name).name!=name or not name:
                raise ValueError("invalid study artifact name")
            source = manifest_path.parent/name
            if source.resolve().parent!=manifest_path.parent.resolve():
                raise ValueError("artifact outside study directory")
            dest = check_write(package/name,source.stat().st_size,operation="保存复核包")
            shutil.copyfile(source,dest)
            if sha(dest)!=sha(source):
                raise ValueError("staged artifact checksum mismatch")
            files.append(dict(name=name,role=entry["role"],retention=entry["retention"],availability="uploaded",
                              sha256=sha(dest),size=dest.stat().st_size))
        final = dict(status=status,summary=summary,artifacts=files)
        submission = dict(run_id=run_id,preflight_id=self.bundle["preflight"]["preflight_id"],final=final,manifest_sha256=digest(final))
        save(package/"submission.json",submission)
        print(f"Desktop upload package ready: {package}",flush=True)
        ack = self.output/f"{self.stage}-receipt.json"
        for tick in range(180):
            if ack.is_file():
                receipt=verify_receipt(read(ack)["receipt"],submission)
                save(manifest_path.parent/"server-receipt.json",receipt)
                return receipt
            if tick%12==0:
                print("Waiting for authenticated desktop upload and readback",flush=True)
            time.sleep(5)
        raise RuntimeError("Package preserved pending authenticated desktop upload")


def launch(stage, entry=ENTRY, output=OUT):
    bundle = read(output/f"{stage}-desktop.json")
    registration, preflight = bundle["run"],bundle["preflight"]
    plan_file, questions = output/f"{stage}-plan.json", output/f"{stage}-questions.json"
    plan = read(plan_file)
    acq, analysis = validate_plan(plan)
    if (not re.fullmatch(r"record-[a-f0-9-]{36}",registration["id"]) or registration["state"]!="running"
            or registration["decision"]!="new_acquisition" or registration["archive_state"]!="pending"
            or registration["plan"]!=plan or registration["plan_sha"]!=digest(plan)
            or registration["preflight_id"]!=preflight["preflight_id"] or preflight["schema"]!="research_test_preflight.v1"
            or preflight["plan_sha256"]!=digest(plan) or preflight["acquisition_key"]!=acq or preflight["analysis_key"]!=analysis
            or any(m["suggestion"] in {"already_running","reuse_results","reuse_acquisition"} for m in preflight["matches"])):
        raise ValueError("desktop round trip did not validate this new acquisition")
    local_name = "run-role-"+registration["id"].removeprefix("record-")
    record = checked_dir(HERE/"result/routine"/local_name, 8*2**20, operation="本机运行记录")
    work = checked_dir(storage_dir("scratch","routine",local_name),2**30,operation="角色复用运行预算")
    if (work/"output.log").exists():
        raise ValueError("The registered run has already started. Reuse its archived work.")
    importer = record/"desktop-registration.json"
    if importer.exists() and read(importer)!=registration:
        raise ValueError("different existing desktop registration")
    save(importer,registration)
    save(record/"preflight.json",preflight)
    save(record/"protocol.json",plan)
    shutil.copyfile(entry,record/"test.py")
    shutil.copyfile(questions,record/"questions.json")
    save(record/"review.json",dict(script=str(entry.relative_to(ROOT)),scratch=str(work),server_run_id=registration["id"]))
    args = argparse.Namespace(archive_plan=plan_file,script=entry,questions=questions,peak_gib=1,archive_decision="new_acquisition",
                              archive_reason=registration["reason"],archive_source_run_id=registration["source_run_id"],
                              model_snapshot=None,tokenizer_sha256=None,seed=None,script_args=[])
    return run(args,archive_client=DesktopDocuments(stage,bundle,output),registered_record=record)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=SIZES)
    raise SystemExit(launch(parser.parse_args().stage))
