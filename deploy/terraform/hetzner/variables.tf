variable "server_name" {
  description = "Name for the preview server."
  type        = string
  default     = "paperintel-preview"
}

variable "server_type" {
  description = "Hetzner Cloud server type. Use a type with at least 4 GB RAM for the bundled preview stack."
  type        = string
  default     = "cx33"
}

variable "location" {
  description = "Hetzner Cloud location, for example nbg1, fsn1, or hel1."
  type        = string
  default     = "nbg1"
}

variable "ssh_public_key" {
  description = "Public SSH key to install for the deploy user. Never provide a private key."
  type        = string

  validation {
    condition     = can(regex("^(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp256) ", var.ssh_public_key))
    error_message = "Provide an OpenSSH public key, not a private key."
  }
}

variable "ssh_allowed_cidrs" {
  description = "IPv4/IPv6 CIDRs allowed to connect to SSH port 22. Keep this restricted to the operator's address."
  type        = list(string)

  validation {
    condition     = length(var.ssh_allowed_cidrs) > 0
    error_message = "At least one restricted SSH source CIDR is required."
  }
}
