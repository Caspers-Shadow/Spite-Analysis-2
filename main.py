import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nogui', action='store_true')
    args = parser.parse_args()
    if args.nogui:
        from ai.training import run_selfplay
        stats = run_selfplay(num_games=100)
        print(stats)
    else:
        # Launch GUI if Qt is available
        try:
            from gui.main_window import MainWindow
            from gui.qt_compat import binding_name
        except Exception as e:
            print("Qt bindings (PyQt6 or PySide6) are not installed or could not be imported.")
            print("On Windows PowerShell try: \n  python -m pip install --user PyQt6\n")
            print("Alternatively install PySide6: python -m pip install --user PySide6")
            print("Error:", e)
            return
        from .gui.qt_compat import QtWidgets
        app = QtWidgets.QApplication([])
        w = MainWindow()
        w.show()
        app.exec()

if __name__ == '__main__':
    main()
