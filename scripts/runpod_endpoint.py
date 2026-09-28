#!/usr/bin/env python3
"""Creates the RunPod serverless endpoint the GPU lessons run on.

Run it once, after the worker image is pullable:

    RUNPOD_API_KEY=... python3 scripts/runpod_endpoint.py

It checks that the image can be pulled anonymously, creates a serverless
template pointing at it, creates an endpoint on the GPU you ask for, and prints
the `RUNPOD_ENDPOINT_ID` line to put in the deployed `.env`. Running it again
reuses the endpoint it already made rather than making a second one.

The image is published by `.github/workflows/publish-gpu-worker.yml` to
`ghcr.io/weights-and-wires/speech-gpu-worker:latest`. A package published from
Actions starts private, and RunPod pulls anonymously, so make the package public
in the organisation's package settings first. `--skip-pull-check` goes ahead
anyway, for when you have added a container registry credential in the RunPod
console instead. `--registry-auth-id <id>` attaches that credential to the
template, and `--registry-auth-id list` prints the ones the account has.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.runpod.io/graphql"

# Cloudflare fronts both api.runpod.io and ghcr.io, and it rejects urllib's
# default user agent with error 1010. Every request here sends its own.
USER_AGENT = "weights-and-wires-speech/1.0"
IMAGE = "ghcr.io/weights-and-wires/speech-gpu-worker:latest"
TEMPLATE_NAME = "speech-gpu-worker"
ENDPOINT_NAME = "speech-course-gpu"

# RunPod's pool ids, from docs.runpod.io/references/gpu-types. The first is the
# one the course is written for; the others accept a job when it is busy.
GPU_POOLS = ("AMPERE_80", "ADA_48_PRO", "ADA_24")


def post(query: str, api_key: str, attempts: int = 3) -> dict:
    """Send one GraphQL request, retrying the statuses RunPod returns in bursts."""
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            f"{API}?api_key={api_key}",
            data=json.dumps({"query": query}).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code in (403, 429, 500, 502, 503) and attempt < attempts:
                time.sleep(2 * attempt)
                continue
            raise SystemExit(
                f"RunPod returned HTTP {exc.code}. {detail}\n"
                "A 403 here usually means the API key is wrong or rate limited."
            ) from None
        except urllib.error.URLError as exc:
            if attempt < attempts:
                time.sleep(2 * attempt)
                continue
            raise SystemExit(f"Could not reach RunPod: {exc.reason}") from None

        if "errors" in body:
            raise SystemExit(f"RunPod rejected the request: {body['errors']}")
        return body["data"]
    raise SystemExit("RunPod did not answer after several attempts.")


def image_is_pullable(image: str) -> tuple[bool, str]:
    """Ask GHCR for an anonymous pull token and then for the manifest."""
    repository, _, tag = image.removeprefix("ghcr.io/").partition(":")
    tag = tag or "latest"
    token_request = urllib.request.Request(
        "https://ghcr.io/token"
        f"?scope=repository:{repository}:pull&service=ghcr.io",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(token_request, timeout=30) as response:
            token = json.loads(response.read()).get("token", "")
    except urllib.error.HTTPError as exc:
        return False, f"GHCR refused an anonymous token: {exc.code}"
    if not token:
        return False, "GHCR issued no anonymous token, so the package is private"

    request = urllib.request.Request(
        f"https://ghcr.io/v2/{repository}/manifests/{tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
            "Accept": ", ".join(
                [
                    "application/vnd.oci.image.index.v1+json",
                    "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.docker.distribution.manifest.v2+json",
                ]
            ),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            return True, "pullable"
    except urllib.error.HTTPError as exc:
        return False, f"GHCR returned {exc.code} for the manifest"


def existing_endpoint(api_key: str) -> dict | None:
    data = post("query { myself { endpoints { id name templateId } } }", api_key)
    for endpoint in data["myself"]["endpoints"] or []:
        if endpoint["name"] == ENDPOINT_NAME:
            return endpoint
    return None


def create_template(api_key: str, disk_gb: int, registry_auth_id: str = "") -> str:
    # A private image needs the id of a container registry credential saved in
    # the RunPod console. A public image needs no credential at all.
    auth = (
        f'containerRegistryAuthId: "{registry_auth_id}"' if registry_auth_id else ""
    )
    query = f"""
    mutation {{
      saveTemplate(input: {{
        name: "{TEMPLATE_NAME}"
        imageName: "{IMAGE}"
        containerDiskInGb: {disk_gb}
        volumeInGb: 0
        isServerless: true
        dockerArgs: ""
        env: []
        {auth}
        readme: "The GPU worker for the speech course lessons marked gpu: true."
      }}) {{ id name imageName }}
    }}
    """
    template = post(query, api_key)["saveTemplate"]
    return template["id"]


def registry_creds(api_key: str) -> list[dict]:
    data = post("query { myself { containerRegistryCreds { id name } } }", api_key)
    return data["myself"]["containerRegistryCreds"] or []


def create_endpoint(
    api_key: str, template_id: str, gpu_ids: str, workers_max: int, idle: int
) -> dict:
    query = f"""
    mutation {{
      saveEndpoint(input: {{
        name: "{ENDPOINT_NAME}"
        templateId: "{template_id}"
        gpuIds: "{gpu_ids}"
        workersMin: 0
        workersMax: {workers_max}
        idleTimeout: {idle}
        scalerType: "QUEUE_DELAY"
        scalerValue: 4
        locations: ""
      }}) {{ id name gpuIds workersMax idleTimeout }}
    }}
    """
    return post(query, api_key)["saveEndpoint"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-ids",
        default=",".join(GPU_POOLS),
        help="RunPod GPU pools, in order of preference: AMPERE_80 is the A100 "
        "80GB, ADA_48_PRO holds the L40 and L40S, and ADA_24 the RTX 4090. "
        "Any of them runs the course; the fallbacks keep a job from queueing "
        "when A100s are scarce.",
    )
    parser.add_argument(
        "--workers-max",
        type=int,
        default=2,
        help="The most workers that can run at once. Workers scale to zero when idle.",
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=10,
        help="Seconds a worker stays warm after a job, before scaling to zero.",
    )
    parser.add_argument("--container-disk-gb", type=int, default=25)
    parser.add_argument(
        "--registry-auth-id",
        default="",
        help="The id of a container registry credential saved in the RunPod "
        "console, for a private image. Pass `list` to print the ones the "
        "account has.",
    )
    parser.add_argument("--skip-pull-check", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print("Set RUNPOD_API_KEY in the environment first.", file=sys.stderr)
        return 2

    if args.registry_auth_id == "list":
        creds = registry_creds(api_key)
        if not creds:
            print("The account has no container registry credentials.")
        for cred in creds:
            print(f"{cred['id']}  {cred['name']}")
        return 0

    # A saved credential is what makes a private image pullable, so the
    # anonymous check does not apply when one is given.
    if not args.skip_pull_check and not args.registry_auth_id:
        ok, why = image_is_pullable(IMAGE)
        if not ok:
            print(f"{IMAGE} is not pullable: {why}.\n", file=sys.stderr)
            print(
                "Make the package public in the organisation's package settings,\n"
                "or add a container registry credential in the RunPod console and\n"
                "run this again with --skip-pull-check.",
                file=sys.stderr,
            )
            return 1
        print(f"{IMAGE} is pullable.")

    found = existing_endpoint(api_key)
    if found:
        print(f"Endpoint {ENDPOINT_NAME} already exists.")
        endpoint_id = found["id"]
    else:
        template_id = create_template(
            api_key, args.container_disk_gb, args.registry_auth_id
        )
        print(f"Created template {template_id}")
        endpoint = create_endpoint(
            api_key,
            template_id,
            args.gpu_ids,
            args.workers_max,
            args.idle_timeout,
        )
        endpoint_id = endpoint["id"]
        print(f"Created endpoint {endpoint_id} on {endpoint['gpuIds']}")

    print("\nPut these two lines in the deployed .env, then restart the app:\n")
    print(f"RUNPOD_API_KEY={api_key[:8]}...")
    print(f"RUNPOD_ENDPOINT_ID={endpoint_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
