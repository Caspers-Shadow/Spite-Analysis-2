import tkinter as tk
from tkinter import messagebox
from game.game_state import GameState
from ai.heuristic_agent import HeuristicAgent
from ai.random_agent import RandomAgent


class TkMainApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Spite Analysis (Tk)')
        self.env = GameState()
        self.agent0 = None
        self.agent1 = None
        self.selected_hand_index = None

        # Layout
        top = tk.Frame(self.root)
        top.pack(fill='x')
        self.opponent_label = tk.Label(top, text='Opponent')
        self.opponent_label.pack(side='left')

        mid = tk.Frame(self.root)
        mid.pack(fill='x')
        # builds
        self.build_buttons = []
        for i in range(4):
            b = tk.Button(mid, text=f'Build {i}\n(empty)', width=20, height=4, command=lambda i=i: self.on_build_click(i))
            b.pack(side='left', padx=4, pady=4)
            self.build_buttons.append(b)

        # hand
        handf = tk.Frame(self.root)
        handf.pack(fill='x')
        self.hand_buttons = []
        for i in range(5):
            hb = tk.Button(handf, text='', width=10, height=4, command=lambda i=i: self.on_hand_click(i))
            hb.pack(side='left', padx=4, pady=4)
            self.hand_buttons.append(hb)

        # discards
        df = tk.Frame(self.root)
        df.pack(fill='x')
        self.discard_buttons = []
        for i in range(4):
            db = tk.Button(df, text=f'D{i}\n(empty)', width=12, height=3, command=lambda i=i: self.on_discard_click(i))
            db.pack(side='left', padx=4, pady=4)
            self.discard_buttons.append(db)

        # controls
        ctl = tk.Frame(self.root)
        ctl.pack(fill='x', pady=6)
        self.hvsa_btn = tk.Button(ctl, text='Human vs AI', command=self.start_human_vs_ai)
        self.hvsa_btn.pack(side='left', padx=4)
        self.aivsa_btn = tk.Button(ctl, text='AI vs AI', command=self.start_ai_vs_ai)
        self.aivsa_btn.pack(side='left', padx=4)
        self.reset_btn = tk.Button(ctl, text='Reset', command=self.reset)
        self.reset_btn.pack(side='left', padx=4)
        self.status_label = tk.Label(ctl, text='')
        self.status_label.pack(side='left', padx=8)

        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        # initial state
        self.env.reset()
        self.update_view()

    def on_close(self):
        if messagebox.askokcancel('Quit', 'Quit Spite Analysis?'):
            self.root.destroy()

    def reset(self):
        self.env.reset()
        self.selected_hand_index = None
        self.update_view()

    def start_human_vs_ai(self):
        self.env.reset()
        self.agent0 = None
        self.agent1 = HeuristicAgent()
        self.selected_hand_index = None
        self.update_view()
        self.status_label.config(text='Human vs AI')

    def start_ai_vs_ai(self):
        self.env.reset()
        self.agent0 = HeuristicAgent()
        self.agent1 = RandomAgent()
        self.selected_hand_index = None
        self.update_view()
        self.status_label.config(text='AI vs AI (running)')
        self.root.after(200, self.ai_loop)

    def ai_loop(self):
        if self.env.is_terminal():
            self.status_label.config(text=f'Game over: winner={self.env.winner}')
            return
        cp = self.env.current_player
        agent = self.agent0 if cp == 0 else self.agent1
        if agent is None:
            return
        action = agent.act(self.env, cp)
        self.env.step(action)
        self.update_view()
        self.root.after(200, self.ai_loop)

    def update_view(self):
        # opponent
        opp = self.env.players[1]
        self.opponent_label.config(text=f'Opponent: stock top {opp.top_stock()} hand {len(opp.hand)}')
        # builds
        for i in range(4):
            b = self.env.builds[i]
            txt = ','.join([f'{c[0].rank}{c[0].suit}:{v}' for c, v in b]) if b else '(empty)'
            self.build_buttons[i].config(text=f'Build {i}\n{txt}', bg='SystemButtonFace')
        # hand
        hs = self.env.players[0].hand
        for i in range(5):
            if i < len(hs):
                self.hand_buttons[i].config(text=str(hs[i]), state='normal')
                if self.selected_hand_index == i:
                    self.hand_buttons[i].config(relief='sunken', bg='yellow')
                else:
                    self.hand_buttons[i].config(relief='raised', bg='SystemButtonFace')
            else:
                self.hand_buttons[i].config(text='', state='disabled')
        # discards
        for i in range(4):
            pile = self.env.players[0].discards[i]
            txt = pile[-1] if pile else '(empty)'
            self.discard_buttons[i].config(text=f'D{i}\n{txt}')
        # status
        self.status_label.config(text=f'P{self.env.current_player} turn')

    def on_hand_click(self, index):
        if index >= len(self.env.players[0].hand):
            return
        if self.selected_hand_index == index:
            self.selected_hand_index = None
        else:
            self.selected_hand_index = index
        self.update_view()

    def on_build_click(self, build_index):
        # if hand selected, play that card to build
        if self.selected_hand_index is not None:
            action = {'type':'PLAY_HAND','hand_index':self.selected_hand_index,'build_index':build_index}
            self.env.step(action)
            self.selected_hand_index = None
            self.update_view()
            return
        # otherwise try stock play
        if self.env.players[0].top_stock():
            action = {'type':'PLAY_STOCK','build_index':build_index}
            self.env.step(action)
            self.update_view()

    def on_discard_click(self, discard_index):
        if self.selected_hand_index is None:
            messagebox.showinfo('Info','Select a hand card to discard')
            return
        action = {'type':'DISCARD','hand_index':self.selected_hand_index,'discard_index':discard_index}
        self.env.step(action)
        self.selected_hand_index = None
        self.update_view()

    def run(self):
        self.root.mainloop()


def main():
    app = TkMainApp()
    app.run()


if __name__ == '__main__':
    main()
