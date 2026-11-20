import sys
import time
import socket
import subprocess
import os
import threading

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtCore import QUrl, Qt


class CustomWebEnginePage(QWebEnginePage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.featurePermissionRequested.connect(self._on_permission_requested)

    def _on_permission_requested(self, security_origin, feature):
        """Grant microphone access; deny everything else."""
        if feature == QWebEnginePage.Feature.MediaAudioCapture:
            print(f"Granting microphone access for: {security_origin.toString()}")
            self.setFeaturePermission(
                security_origin, feature, QWebEnginePage.PermissionPolicy.PermissionGrantedByUser
            )
        else:
            self.setFeaturePermission(
                security_origin, feature, QWebEnginePage.PermissionPolicy.PermissionDeniedByUser
            )


class BrowserWindow(QMainWindow):
    def __init__(self, url: str, node_process: subprocess.Popen):
        super().__init__()
        self.node_process = node_process

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        self.view = QWebEngineView(self)
        self.view.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.view.page().setBackgroundColor(Qt.GlobalColor.transparent)

        custom_page = CustomWebEnginePage(self.view)
        self.view.setPage(custom_page)
        self.setCentralWidget(self.view)
        self.view.load(QUrl(url))

        self.setWindowTitle("Emergency Assistant")
        self.resize(1000, 600)

    def closeEvent(self, event):
        self.node_process.terminate()
        self.node_process.wait()
        super().closeEvent(event)


class GUIApp:
    """
    Starts a Node.js dev server and displays the app inside a PySide6 window.
    """

    def __init__(
        self,
        node_cmd: list = None,
        node_cwd: str = "GUI",
        url: str = "http://localhost:8080/",
        port: int = 8080,
    ):
        self.node_cmd  = node_cmd or ["npm", "run", "dev"]
        self.node_cwd  = node_cwd
        self.url       = url
        self.port      = port
        self.node_process = None

    # ------------------------------------------------------------------

    def _is_server_ready(self) -> bool:
        try:
            with socket.create_connection(("localhost", self.port), timeout=5):
                return True
        except (socket.timeout, ConnectionRefusedError):
            return False

    def _stream_reader(self, pipe, tag: str) -> None:
        """Read and print subprocess pipe output line by line."""
        for line in iter(pipe.readline, ""):
            print(f"[{tag}] {line}", end="")
        pipe.close()

    def _start_node_server(self) -> None:
        env = os.environ.copy()
        env["BROWSER"] = "none"  # prevent Node from auto-opening a browser

        self.node_process = subprocess.Popen(
            self.node_cmd,
            cwd=self.node_cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        threading.Thread(
            target=self._stream_reader,
            args=(self.node_process.stdout, "STDOUT"),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._stream_reader,
            args=(self.node_process.stderr, "STDERR"),
            daemon=True,
        ).start()

        print("Waiting for Node server to start…")
        while not self._is_server_ready():
            time.sleep(1)
        print("Node server is ready.")

    # ------------------------------------------------------------------

    def run(self) -> None:
        self._start_node_server()
        app    = QApplication(sys.argv)
        window = BrowserWindow(self.url, self.node_process)
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    gui_app = GUIApp()
    gui_app.run()
