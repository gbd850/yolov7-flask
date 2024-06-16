from flask import Flask, flash, request, redirect, render_template, send_from_directory
from werkzeug.utils import secure_filename
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = ROOT_DIR + '\\uploads'
ALLOWED_EXTENSIONS = {'jpg', 'mp4'}

sys.path.append(ROOT_DIR + '\\segment')
from segment.predict import main, parse_opt

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            opt = parse_opt()
            opt.source = 'uploads\\' + filename
            opt.weights = ['Weights\\best.pt']
            result_dir = main(opt)
            return send_from_directory(result_dir, filename, as_attachment=True)
        flash('Selected file with forbidden extension')
        return redirect(request.url)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    # opt = parse_opt()
    # opt.source = 'Image\\image.jpg'
    # opt.weights = ['Weights\\best.pt']
    # main(opt)
