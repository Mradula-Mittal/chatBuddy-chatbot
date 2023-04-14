from flask import Flask, request, render_template, jsonify
import processor

app = Flask(__name__)

app.config['SECRET_KEY'] = "kcmkadsckmscadskmcasdc"

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', **locals())

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbotResponse():

    if request.method == 'POST':
        the_question = request.form['question']
        print(the_question)
        response = processor.chatbot_response(the_question)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888', debug= True)