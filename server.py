"""Utilities for managing a remote server over SSH/SFTP.

This module wraps :mod:`paramiko` and exposes a small set of convenience
methods for connecting to a remote host, transferring files and executing
commands.  Configuration can be loaded from a YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import os
import stat

import paramiko
from tqdm import tqdm

from utils.lib_io import read_yaml_file


@dataclass
class Server:
    """Simple wrapper around :class:`paramiko.SSHClient`.

    Parameters
    ----------
    hostname:
        IP or hostname of the remote machine.
    username:
        Login user name.
    password:
        Password for ``username``.
    port:
        SSH port, defaults to ``22``.
    use_sftp:
        Whether to initialise an SFTP client.  Enabled by default.
    """

    hostname: str
    username: str
    password: str
    port: int = 22
    use_sftp: bool = True
    client: paramiko.SSHClient = field(init=False)
    sftp: Optional[paramiko.SFTPClient] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.connect()

    @classmethod
    def from_yaml(cls, cfg_path: str = "cfg/cfg_server.yaml") -> "Server":
        """Construct :class:`Server` from a YAML configuration file."""
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        return cls(
            hostname=cfg.hostname,
            username=cfg.username,
            password=cfg.password,
            port=int(getattr(cfg, "port", 22)),
            use_sftp=bool(getattr(cfg, "use_sftp", True)),
        )

    def __str__(self) -> str:  # pragma: no cover - trivial
        return (
            f"[Server]: hostname={self.hostname}, port={self.port}, "
            f"username={self.username}, use_sftp={self.use_sftp}"
        )

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """Establish the SSH (and optional SFTP) connection."""
        print("==========\nServer Connecting...")
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.client.connect(
                hostname=self.hostname,
                port=self.port,
                username=self.username,
                password=self.password,
            )
            if self.use_sftp:
                self.sftp = self.client.open_sftp()
            print("Server Connected\n==========")
        except paramiko.AuthenticationException:
            print("Authentication failed. Please check your credentials.")
        except paramiko.SSHException as ssh_ex:
            print("Error occurred while connecting or establishing an SSH session:", ssh_ex)
        except paramiko.ssh_exception.NoValidConnectionsError as conn_ex:
            print("Unable to connect to the server:", conn_ex)
        except Exception as ex:  # pragma: no cover - defensive
            print("An unexpected error occurred:", ex)

    def disconnect(self) -> None:
        """Close the SSH/SFTP connections."""
        self.client.close()
        if self.sftp is not None:
            self.sftp.close()

    # ------------------------------------------------------------------ #
    # File operations
    # ------------------------------------------------------------------ #
    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        *,
        verbose: bool = False,
        show_progress: bool = False,
    ) -> None:
        """Upload a local file to the remote server.

        Parameters
        ----------
        local_path:
            Path to the local file.
        remote_path:
            Destination path on the remote host.
        verbose:
            Print a confirmation message.
        show_progress:
            Display a progress bar while uploading.
        """
        if self.sftp is None:
            raise RuntimeError("SFTP client not initialised.")

        if show_progress:
            file_size = os.path.getsize(local_path)

            def _callback(transferred: int, _total: int) -> None:
                progress.update(transferred - progress.n)

            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Upload {os.path.basename(local_path)}",
            ) as progress:
                self.sftp.put(local_path, remote_path, callback=_callback)
        else:
            self.sftp.put(local_path, remote_path)

        if verbose:
            print(f"{local_path} has been transferred to {remote_path}")

    def download_file(self, remote_path: str, local_path: str, *, verbose: bool = False) -> None:
        """Download a file from the remote server."""
        if self.sftp is None:
            raise RuntimeError("SFTP client not initialised.")
        self.sftp.get(remote_path, local_path)
        if verbose:
            print(f"{remote_path} has been transferred to {local_path}")

    def download_directory(self, remote_dir: str, local_dir: str, *, verbose: bool = False) -> None:
        """Recursively download a directory from the remote server."""
        if self.sftp is None:
            raise RuntimeError("SFTP client not initialised.")

        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        for entry in self.sftp.listdir(remote_dir):
            remote_path = os.path.join(remote_dir, entry)
            local_path = os.path.join(local_dir, entry)
            if stat.S_ISDIR(self.sftp.stat(remote_path).st_mode):
                self.download_directory(remote_path, local_path)
            else:
                self.sftp.get(remote_path, local_path)

        if verbose:
            print(f"{remote_dir} has been transferred to {local_dir}")

    def create_remote_file(self, content: str, remote_path: str, *, verbose: bool = False) -> None:
        """Create a file on the remote server with the given content."""
        if self.sftp is None:
            raise RuntimeError("SFTP client not initialised.")
        with self.sftp.open(remote_path, "w") as remote_file:
            remote_file.write(content)
        if verbose:
            print(f"{remote_path} has been created")

    # ------------------------------------------------------------------ #
    # Command execution
    # ------------------------------------------------------------------ #
    def execute_command(self, command: str, *, verbose: bool = False) -> None:
        """Execute a shell command on the remote server."""
        stdin, stdout, stderr = self.client.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()  # wait for execution
        if exit_status == 0:
            if verbose:
                print("Successfully executed command on server")
        else:
            print(f"Failed to execute command. Error code: {exit_status}")
            error_output = stderr.read().decode().strip()
            print(f"Error output: {error_output}")
        if verbose:
            output = stdout.read().decode("utf-8")
            print(f"Server output:\n{output}")


def main() -> None:  # pragma: no cover - example usage
    server = Server.from_yaml("cfg/cfg_server.yaml")
    print(server)
    server.execute_command("ls", verbose=True)
    server.disconnect()


if __name__ == "__main__":
    main()

