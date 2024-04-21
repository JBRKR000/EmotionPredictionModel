import sys
from PyQt5.QtWidgets import QApplication
from user_interface import UserInterface

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = UserInterface()
    ui.show()
    sys.exit(app.exec_())
