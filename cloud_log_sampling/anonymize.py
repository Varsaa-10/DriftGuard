import argparse
import json
import random
import re
import string
import zipfile
from pathlib import Path

IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
ACCOUNT_ID_RE = re.compile(r"\d{12}")


class Anonymizer:
    """Consistently replaces identity values across all records.

    Same original value -> always the same fake value.
    Different original values -> always different fake values.
    """

    def __init__(self):
        self.user_map = {}  # userName -> "userN"
        self.value_map = {}  # (field, original_value) -> fake value
        self.used_values = {}  # field -> set of fake values already handed out

    def anon_user_name(self, name):
        if name not in self.user_map:
            self.user_map[name] = f"user{len(self.user_map) + 1}"
        return self.user_map[name]

    def anon_value(self, field, value, generator):
        key = (field, value)
        if key not in self.value_map:
            used = self.used_values.setdefault(field, set())
            fake = generator()
            while fake in used:
                fake = generator()
            used.add(fake)
            self.value_map[key] = fake
        return self.value_map[key]

    def anon_account_id(self, value):
        return self.anon_value("accountId", value, lambda: "".join(random.choices(string.digits, k=12)))

    def anon_principal_id(self, value):
        prefix = value[:4] if value[:4].isalpha() else "AIDA"
        return self.anon_value(
            "principalId",
            value,
            lambda: prefix + "".join(random.choices(string.ascii_uppercase + string.digits, k=17)),
        )

    def anon_access_key_id(self, value):
        prefix = value[:4] if value[:4].isalpha() else "AKIA"
        return self.anon_value(
            "accessKeyId",
            value,
            lambda: prefix + "".join(random.choices(string.ascii_uppercase + string.digits, k=16)),
        )

    def anon_ip_address(self, value):
        octets = value.split(".")
        prefix = ".".join(octets[:2])
        return self.anon_value(
            "sourceIPAddress",
            value,
            lambda: f"{prefix}.{random.randint(0, 255)}.{random.randint(0, 255)}",
        )


def substitute(text: str, replacements: dict) -> str:
    for orig, anon in replacements.items():
        if orig in text:
            text = text.replace(orig, anon)
    return text


def anonymize_identity(identity: dict, anonymizer: Anonymizer) -> None:
    if not isinstance(identity, dict):
        return

    replacements = {}

    if identity.get("accountId"):
        orig = identity["accountId"]
        anon = anonymizer.anon_account_id(orig)
        identity["accountId"] = anon
        replacements[orig] = anon

    if identity.get("principalId"):
        orig = identity["principalId"]
        anon = anonymizer.anon_principal_id(orig)
        identity["principalId"] = anon
        replacements[orig] = anon

    if identity.get("accessKeyId"):
        orig = identity["accessKeyId"]
        anon = anonymizer.anon_access_key_id(orig)
        identity["accessKeyId"] = anon
        replacements[orig] = anon

    if identity.get("userName"):
        orig = identity["userName"]
        if identity.get("type") == "IAMUser":
            # Only real human IAM users get replaced with userN.
            anon = anonymizer.anon_user_name(orig)
            identity["userName"] = anon
            replacements[orig] = anon
        else:
            # AWS service-linked / infra roles keep their descriptive name,
            # but any embedded accountId/principalId/accessKeyId gets swapped.
            identity["userName"] = substitute(orig, replacements)

    if identity.get("arn"):
        identity["arn"] = substitute(identity["arn"], replacements)

    session_issuer = identity.get("sessionContext", {}).get("sessionIssuer")
    if session_issuer:
        anonymize_identity(session_issuer, anonymizer)


def anonymize_response_elements(response_elements: dict, anonymizer: Anonymizer) -> None:
    if not isinstance(response_elements, dict):
        return

    credentials = response_elements.get("credentials")
    if isinstance(credentials, dict) and credentials.get("accessKeyId"):
        credentials["accessKeyId"] = anonymizer.anon_access_key_id(credentials["accessKeyId"])

    federated_user = response_elements.get("federatedUser")
    if isinstance(federated_user, dict):
        fed_id = federated_user.get("federatedUserId")
        if fed_id and ":" in fed_id:
            acct, _, rest = fed_id.partition(":")
            federated_user["federatedUserId"] = f"{anonymizer.anon_account_id(acct)}:{rest}"
        arn = federated_user.get("arn")
        if arn:
            for acct in ACCOUNT_ID_RE.findall(arn):
                arn = arn.replace(acct, anonymizer.anon_account_id(acct))
            federated_user["arn"] = arn


def sweep_known_values(obj, anonymizer: Anonymizer):
    """Safety net: replace any already-seen real identity value wherever it resurfaces
    (bucket names, resource ARNs, policy text, hostnames, etc.)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = sweep_known_values(value, anonymizer)
        return obj
    if isinstance(obj, list):
        return [sweep_known_values(item, anonymizer) for item in obj]
    if isinstance(obj, str):
        for (field, orig), anon in anonymizer.value_map.items():
            if orig in obj:
                obj = obj.replace(orig, anon)
        for orig, anon in anonymizer.user_map.items():
            if orig in obj:
                obj = obj.replace(orig, anon)
        return obj
    return obj


def anonymize_records(records: list) -> list:
    anonymizer = Anonymizer()
    for record in records:
        user_identity = record.get("userIdentity")
        if user_identity:
            anonymize_identity(user_identity, anonymizer)

        source_ip = record.get("sourceIPAddress")
        if source_ip and IPV4_RE.match(source_ip):
            record["sourceIPAddress"] = anonymizer.anon_ip_address(source_ip)

        recipient_account_id = record.get("recipientAccountId")
        if recipient_account_id:
            record["recipientAccountId"] = anonymizer.anon_account_id(recipient_account_id)

        anonymize_response_elements(record.get("responseElements"), anonymizer)

    for record in records:
        sweep_known_values(record, anonymizer)

    print(anonymizer.user_map)
    print(anonymizer.value_map)
    print(anonymizer.used_values)
    return records


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    return extract_to


def merge_cloudtrail_logs(root: Path) -> dict:
    merged_records = []
    json_paths = sorted(root.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=False)
    for json_path in json_paths:
        print(f"Processing {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("Records", [])
        merged_records.extend(records)
    return {"Records": merged_records}


def main():
    parser = argparse.ArgumentParser(description="Merge CloudTrail JSON logs from a zip file into one file.")
    parser.add_argument("zip_file", type=Path, help="Path to the zip file containing CloudTrail JSON logs")
    parser.add_argument("-o", "--output", type=Path, default=Path("merged_cloudtrail.json"), help="Output path for the merged JSON file")
    args = parser.parse_args()

    extract_dir = args.zip_file.with_suffix("")
    extract_zip(args.zip_file, extract_dir)

    merged = merge_cloudtrail_logs(extract_dir)
    merged["Records"] = anonymize_records(merged["Records"])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(merged['Records'])} records into {args.output}")


if __name__ == "__main__":
    main()
