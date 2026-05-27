import sys
import os
from flask import Flask, jsonify, request, send_from_directory

# Ensure repo root is on sys.path so top-level packages import correctly when
# running this file as a script (python web/app.py)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from game.game_state import GameState
from ai.heuristic_agent import HeuristicAgent
from ai.random_agent import RandomAgent
import threading

app = Flask(__name__, static_folder='static')

# Global env for simple demo (single session)
ENV = GameState()
AGENTS = [None, HeuristicAgent()]


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/state')
def state():
    return jsonify(ENV.get_state())


@app.route('/reset', methods=['POST'])
def reset():
    ENV.reset()
    return jsonify({'status':'ok','state':ENV.get_state()})


@app.route('/action', methods=['POST'])
def action():
    payload = request.json
    if not payload:
        return jsonify({'error':'no payload'}), 400
    state, done, info = ENV.step(payload)
    # if AI turn, auto-step
    while not ENV.is_terminal() and ENV.current_player != 0 and AGENTS[ENV.current_player] is not None:
        act = AGENTS[ENV.current_player].act(ENV, ENV.current_player)
        ENV.step(act)
    return jsonify({'state':ENV.get_state(), 'done':ENV.is_terminal(), 'info':info})


def run_app():
    app.run(port=5000, debug=False)

if __name__ == '__main__':
    ENV.reset()
    run_app()
