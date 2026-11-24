# Deploy the public demo

Render builds the same Dockerfile used by the local Compose stack. The public
beta uses the statistical engine for the bundled demo and for synthetic or
de-identified CSV uploads. Both paths provide the web report and A4 PDF. The
local Compose stack also supports OpenAI-assisted audits and MLflow tracking.

## Connect Render to GitHub

1. Sign in to [Render](https://dashboard.render.com/) with GitHub and grant Render
   access to the `AhmadAssi1/healthequity-agent` repository.
2. Select **New → Blueprint**, choose the repository and its `main` branch, and
   apply the settings from `render.yaml`. The Blueprint uses the Free instance
   plan and enables automatic deployments after CI checks pass.
3. Wait for the first deployment to become live. Open the service URL and check
   `/_stcore/health`. On **Demo**, click **Run demo audit** and open the downloaded
   A4 PDF. Try **Audit** with a synthetic CSV as well. Ask someone outside the
   project to run the demo and download its PDF within five minutes.

The Free instance has 512 MB of memory and can go to sleep when idle. The first
visit after a period of inactivity may need to wait for the service to start.

[Render Blueprints](https://render.com/docs/blueprint-spec) ·
[Free instance limits](https://render.com/docs/free)
