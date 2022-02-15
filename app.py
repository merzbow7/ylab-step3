import base64
from pathlib import Path

from flask import Flask, render_template, request

from geometry import figure

app = Flask(__name__)


def einstein() -> str:
    img = Path(__file__).parent / 'einstein.jpg'
    b64 = base64.b64encode(img.read_bytes())
    pic_hash = b64.decode('utf-8')
    return pic_hash


def is_negative(arg):
    return 'Must be more than 0' if arg <= 0 else ''


def check_angle(angle):
    return 0 <= angle <= 180


def check_params(params: dict):
    errors = {key: is_negative(value) for key, value in params.items()}
    if not check_angle(params.get('angle', 45)):
        errors['angle'] = 'Angle must be 0 < \u03B1 < 180'
    a = params.get('a', 45)
    c = params.get('c', a)
    if a <= c and 'c' in params:
        errors['c'] = 'Must be less than a'
    return errors


@app.route('/')
def index() -> str:
    cls_name = request.args.get('figure', default='Circle', type=str)
    figure_class = getattr(figure, cls_name)
    slots = figure_class.__slots__
    class_args = {attr: request.args.get(attr, type=float, default=25)
                  for attr in slots}
    shape = figure_class(**class_args)
    summary = shape.get_summary()
    errors = check_params(class_args)
    if not any(errors.values()):
        img = shape.plot()
    else:
        img = einstein()
        summary = 'Error!'
    slots = {slot: class_args[slot] for slot in slots}
    return render_template('index.html',
                           img=img,
                           slots=slots,
                           errors=errors,
                           choose=cls_name,
                           summary=summary,
                           figures=figure.__all__)


def main(*args, **kwargs):
    print(f'{args=}')
    print(f'{kwargs=}')
    app.run(debug=False)

