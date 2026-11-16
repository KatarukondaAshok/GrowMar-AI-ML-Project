# Hosting decision (Task 17)
Free-plan limits change often. The facts below were read from the providers' own documentation and review articles on 16 Nov 2026. **Check the pricing page again on the day you deploy.**

## What the service needs (measured)
API process about 198 MB + mock HR process about 90 MB = about **290 MB RAM**, 7 models, start-up 0.2 s once loaded, Docker image on Python 3.12.

## Comparison
| Option | Memory | Sleep behaviour | Build limits | Price per month |
|---|---|---|---|---|
| **Render, Free web service** | 512 MB RAM, 0.1 CPU | Sleeps after 15 min with no traffic; the next request waits about 1 minute | 500 build minutes a month, 750 free instance hours | $0 |
| Render, Starter | 512 MB RAM, 0.5 CPU | Never sleeps | Same build limits | about $7 (check for any workspace fee) |
| Render, Standard | 2 GB RAM, 1 CPU | Never sleeps | Same | about $25 |
| Hugging Face Spaces, CPU Basic | 16 GB RAM, 2 vCPU | Sleeps after 48 h inactive | Docker Space | Hardware is free, **but HF's docs now say creating a Docker or Gradio Space needs a paid plan (PRO)**; only Static Spaces are free. Verify on your account |

Sources: Render docs and pricing pages (free: 512 MB, 0.1 CPU, 15-minute spin-down); Hugging Face Hub documentation, "Spaces overview" and "Spaces hardware".

## Decision
**Deploy the API on Render's Free web service for the review.** Reasons: it costs nothing, 290 MB fits in 512 MB, it builds straight from the `Dockerfile` with `render.yaml`, and the free tier needs no credit card.
**Known cost of this choice:** after 15 minutes idle the first call takes about a minute. The smoke test and the console both use a 90-second timeout for this reason, and the console shows a friendly "waking up" message. Before your demo video, open `/v1/health` once so the service is awake.
**If real users depend on it:** move to Starter (about $7 a month) so it never sleeps. A pause would hurt real users, and the price is small. Write that decision in your report.

## Deploy steps (Render)
1. Push this repo to GitHub (done for Phases 1 to 3; `phase-4/` next).
2. render.com: New, Blueprint, pick the repo. It reads `phase-4/render.yaml` (a copy sits in the repo root; see README).
3. In the service's Environment tab set `AURA_API_KEY` to a long random value (do not put it in the repo).
4. Wait for the build; open `https://YOUR-SERVICE.onrender.com/v1/health`.
5. `AURA_API_KEY=... python smoke_test.py https://YOUR-SERVICE.onrender.com`.
