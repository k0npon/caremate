# Deploying the CareMate perception service on AWS (EC2 + Docker)

What runs on AWS: the OpenCV 5 perception service (`caremate/service/app.py`):
ALIKED features, BF+mutual matching, USAC/MAGSAC verification, look-alike check.
What stays on the robot/laptop: MuJoCo simulation, planning (next-best-view), control.

## 1. Launch the instance (EC2 console, region eu-north-1 Stockholm)
- AMI: Ubuntu Server 26.04 LTS (x86_64). Instance type: c7i-flex.large (2 vCPU, 4 GiB, free-tier eligible) or t3.small.
  (An arm64 Graviton t4g.medium also works: the OpenCV 5.0.0 contrib wheel exists for aarch64.)
- Key pair: create one, download the `.pem`.
- Network: allow SSH (22) from *My IP*; tick "Allow HTTP traffic from the internet" (the container is published on port 80).
- Storage: 20 GiB gp3.

## 2. Copy the service and start it
On the laptop:
```bash
bash scripts/pack_service.sh
chmod 400 ~/Downloads/caremate-key.pem
scp -i ~/Downloads/caremate-key.pem caremate_service.tar.gz ubuntu@<PUBLIC_IP>:~
ssh -i ~/Downloads/caremate-key.pem ubuntu@<PUBLIC_IP>
```
On the instance:
```bash
mkdir -p caremate && tar xzf caremate_service.tar.gz -C caremate && cd caremate
bash deploy/ec2_bootstrap.sh            # prints the API token at the end
# PUBLIC=1 bash deploy/ec2_bootstrap.sh # judge-accessible without a token
```

## 3. Use it
- Browser: `http://<PUBLIC_IP>/` (upload a frame, pick a target).
- Agent: `CAREMATE_API_TOKEN=<token> python -m caremate.agent_loop_v0_7 --target crunch_peanut --backend-url http://<PUBLIC_IP>`
- Local vs cloud benchmark: `python -m caremate.bench_remote_v0_7 --backend-url http://<PUBLIC_IP>`

## Responsible delivery
- Images are processed in memory and never stored; logs contain only target, status, latency and image hash.
- Optional bearer token; upload and image size capped; non-root container with CPU/memory limits and log rotation.
- Cost: a few US cents per hour, charged to credits. Stop the instance when not demonstrating (the public IP can change on restart).
- Plain HTTP for the demo; put a load balancer or CloudFront with TLS in front for anything beyond a demo.
