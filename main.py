import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nogui', action='store_true')
    parser.add_argument('--tk', action='store_true', help='Use Tkinter GUI instead of Qt')
    args = parser.parse_args()
    if args.nogui:
        from ai.training import run_selfplay
        stats = run_selfplay(num_games=100)
        print(stats)
    else:
        if args.tk:
            # Run Tkinter GUI
            try:
                from gui.tk_main import main as tkmain
            except Exception as e:
                print('Tkinter GUI not available or failed to import:', e)
                return
            tkmain()
            return

        # Launch GUI if Qt is available
        try:
            from gui.main_window import MainWindow
            from gui.qt_compat import binding_name
        except Exception as e:
            print("Qt bindings (PyQt6 or PySide6) are not installed or could not be imported.")
            print("To use the Tk fallback run: python main.py --tk")
            print("On Windows PowerShell try: \n  python -m pip install --user PyQt6\n")
            print("Alternatively install PySide6: python -m pip install --user PySide6")
            print("Error:", e)
            return

        from gui.qt_compat import QtWidgets
        app = QtWidgets.QApplication([])
        w = MainWindow()
        w.show()
        app.exec()

if __name__ == '__main__':
    main()
