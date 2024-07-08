import os
import sys
from pose import ui


if __name__ == '__main__':
    app = ui.QApplication(sys.argv)
    num_array = ['1', '2.1', '2.2', '3.1', '3.2', '4', '5.1', '5.2', '6', '7.1', '7.2', '8']
    window = ui.MainWindow(num_array)
    window.show()
    sys.exit(app.exec_())