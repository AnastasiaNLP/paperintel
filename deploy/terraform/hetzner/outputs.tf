output "server_ipv4" {
  description = "Set the preview hostname's DNS A record to this address."
  value       = hcloud_server.preview.ipv4_address
}

output "ssh_command" {
  description = "SSH command for the restricted deploy user."
  value       = "ssh paperintel@${hcloud_server.preview.ipv4_address}"
}
