# Hetzner preview server

This Terraform root creates one Ubuntu 24.04 server, installs Docker and the
Compose plugin, imports a deploy SSH public key, and applies a firewall that
exposes only SSH (restricted to configured CIDRs), HTTP, HTTPS, and ICMP. It
does not create paid resources until `terraform apply` is run.

The project has not declared a cloud provider, so this is the default preview
target. If the preview must use another provider or an already existing VPS,
adapt this small infrastructure layer before applying it.

## Provision the server

Install Terraform 1.6 or newer and create a Hetzner Cloud API token with
permissions to manage servers, firewalls, and SSH keys. Set it only in the
process environment:

```bash
export HCLOUD_TOKEN="<Hetzner API token>"
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`: set the operator's public SSH key and replace the
example SSH CIDR with the operator's current public IP in `/32` or `/128`
notation. Choose the server type and location if needed. Do not commit
`terraform.tfvars` or Terraform state; both are ignored by Git.

Then review and apply:

```bash
terraform init
terraform fmt -check
terraform validate
terraform plan
terraform apply
terraform output
```

The default `cx33` size is a starting point for a single host running the API,
worker, Postgres, Qdrant, and MinIO. Hetzner currently lists CX33 with 4 vCPU,
8 GB RAM, and 80 GB NVMe; its Germany/Finland price is $9.99/month before VAT
and IPv4 charges. This config enables Hetzner's daily server backups, priced at
20% of the server price (seven restore points), for an estimated $11.99/month
before VAT and IPv4 charges. Check [current prices and availability](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/)
and [backup billing and behavior](https://docs.hetzner.com/cloud/billing/faq/)
before applying. Terraform state is local in this starter, so keep a private
backup and use a remote state backend before multiple operators share the
infrastructure.

## Attach the domain and start PaperIntel

Create an `A` record for the preview hostname pointing to
`terraform output -raw server_ipv4`. Wait until public DNS resolves to that
address. Then SSH to the server and clone the repository using your normal
repository access:

```bash
terraform output -raw ssh_command
```

After connecting, wait for cloud-init to finish with `sudo cloud-init status
--wait`. Clone the pushed project revision and enter its directory. Run the
bootstrap script; it prompts for the hostname and provider keys without echoing
the keys, generates random app/database/storage passwords, writes a mode-600
`.env`, builds the images, applies migrations, and waits for HTTPS health:

```bash
chmod +x deploy/preview/bootstrap.sh
deploy/preview/bootstrap.sh
```

Keep `.env` on the server only. The script reuses an existing `.env` on
retries and does not overwrite its secrets. To retrieve the preview token for
testers, run this on the server and share its output through a private channel:

```bash
grep '^PAPERINTEL_API_AUTH_TOKEN=' .env
```

Caddy obtains and renews the TLS certificate when the domain resolves and
ports 80/443 are reachable. Run the smoke checks in
[`PREVIEW_DEPLOYMENT.md`](../../../docs/PREVIEW_DEPLOYMENT.md). Do not expose
ports 5432, 6333, 6334, 9000, or 9001 in the Hetzner firewall.

Hetzner's daily backups cover the whole server disk, but they are not an
application-level export and this repo has not tested a restore. Keep an
independent backup of important data and Terraform state. The preview also has
no managed database, per-person identity, or per-user quotas; keep it for a
small tester group.
