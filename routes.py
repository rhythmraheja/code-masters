from check_sim import *
import subprocess
#from check_sim import *
from IPython.utils.capture import capture_output
from flask import render_template, request, redirect, url_for, flash, session, make_response, Response
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from subprocess import Popen, PIPE, run
from sqlalchemy.sql import func
from wtforms.validators import optional

from forms import StudentSignUpForm, StudentLoginForm, TeacherLoginForm, TeacherSignUpForm
from models import Student, Teacher, Assignment, Question, Testcase, Submission
from app import app, db, bcrypt
from datetime import datetime
from io import StringIO
import os
import csv


# Configuration for file upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'py'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'  

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@login_manager.user_loader
def load_student(student_id):
    return Student.query.get(int(student_id))

@app.route('/')
def index():
    return render_template('index.html', title="Online Assignment Evaluator")

@app.route('/code_editor/<int:question_id>/<int:assignment_id>')
def code_editor(question_id, assignment_id):
    return render_template('code_editor.html', question_id=question_id, assignment_id=assignment_id)




@app.route('/student_signup', methods=['GET', 'POST'])
def student_signup():
    form = StudentSignUpForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        student = Student(
            name=form.name.data,
            roll=form.roll.data,
            exam_roll=form.exam_roll.data,
            email_id=form.email_id.data,
            group=form.group.data,
            password=hashed_password
        )
        db.session.add(student)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('student_signup.html', form=form)

@app.route('/teacher_signup', methods=['GET', 'POST'])
def teacher_signup():
    form = TeacherSignUpForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        teacher = Teacher(
            name=form.name.data,
            email=form.email_id.data,
            password=hashed_password
        )
        db.session.add(teacher)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('teacher_signup.html', form=form)


@app.route('/student_login', methods=['GET', 'POST'])
def student_login():
    form = StudentLoginForm()
    if form.validate_on_submit():
        student = Student.query.filter_by(email_id=form.email_id.data).first()
        if student and bcrypt.check_password_hash(student.password, form.password.data):
            login_user(student)
            # Store the student's ID in the session
            session['student_id'] = student.id
            

            flash('Login successful!', 'success')
            return redirect(url_for('student_dashboard', student_id=student.id))  # Redirect to student dashboard after login
        else:
            flash('Login unsuccessful. Please check email and password.', 'danger')
    return render_template('student_login.html', form=form)


@app.route('/teacher_login', methods=['GET', 'POST'])
def teacher_login():
    form = TeacherLoginForm()
    if form.validate_on_submit():
        teacher = Teacher.query.filter_by(email=form.email.data).first()
        if teacher and bcrypt.check_password_hash(teacher.password, form.password.data):
            flash('Login successful!', 'success')
            session['teacher_id'] = teacher.id
            return redirect(url_for('teacher_dashboard'))  # Redirect to teacher dashboard after login
        else:
            flash('Login unsuccessful. Please check email and password.', 'danger')
    return render_template('teacher_login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    # Check if a student is logged in
    # if 'student_id' in session:
    #     session.pop('student_id', None)  # Remove the student_id from session
    #     flash('You have been logged out as a student.', 'success')
    #
    # # Check if a teacher is logged in
    # elif 'teacher_id' in session:
    #     session.pop('teacher_id', None)  # Remove the teacher_id from session
    #     flash('You have been logged out as a teacher.', 'success')
    # else:
    #     flash('No user is currently logged in.', 'warning')
    logout_user()
    flash('You have been logged out.', 'success')
    # Redirect to login page after logout
    return redirect(url_for('index'))

@app.route('/student_dashboard/<int:student_id>')
@login_required
def student_dashboard(student_id):
    if 'student_id' not in session:
        flash('You must be logged in to submit your work.')
        return redirect(url_for('student_login'))
    elif session['student_id'] != student_id:
        flash('You must be logged in to submit your work.')
        return redirect(url_for('student_login'))
    else:
        student_data = Student.query.filter_by(id=student_id).first()
        group = student_data.group
        current_date = datetime.today().strftime('%Y-%m-%d')
        
        if group[-1] == '1':
            assignments = Assignment.query.filter(Assignment.date1 <= current_date).all()
            student_group = 1
        else:
            assignments = Assignment.query.filter(Assignment.date2 <= current_date).all()
            student_group = 2
        return render_template('student_dashboard.html',
                               assignments=assignments, student_data=student_data, student_group=student_group)

@app.route('/teacher_dashboard')
def teacher_dashboard():
    if 'teacher_id' not in session:
        flash('You must be logged in to submit your work.')
        return redirect(url_for('teacher_login'))
    
    assignments = Assignment.query.all()  
    return render_template('teacher_dashboard.html', assignments=assignments)


@app.route('/create_assignment', methods=['GET', 'POST'])
def create_assignment():
    if request.method == 'POST':
        if 'teacher_id' not in session:
            flash('You must be logged in to submit your work.')
            return redirect(url_for('teacher_login'))
        topic = request.form['topic']
        date1 = request.form['date1']
        date2 = request.form['date2']
        due_date1 = request.form['due_date1']
        due_date2 = request.form['due_date2']
        total_marks = request.form['total_marks']
        num_questions = request.form['num_questions']

        # Create an Assignment object
        new_assignment = Assignment(
            topic=topic,
            date1=date1,
            date2=date2,
            due_date1=due_date1,
            due_date2=due_date2,
            total_marks=total_marks,
            num_questions=num_questions
        )
        t=request.form['topic']
        

        db.session.add(new_assignment)
        db.session.commit()
        folder_path = r'C:\Users\Rhythm\OneDrive\Desktop\nn\IDE for similarity\IDE for similarity\Online_Programming_Assignment_Portal-main (2)\Online_Programming_Assignment_Portal-main\Online_Programming_Assignment_Portal-main\src\\'+t
        create_new_folder(folder_path)



        flash('Assignment created successfully!', 'success')
        # Redirect to the add questions page with the new assignment ID
        return redirect(url_for('add_questions', assignment_id=new_assignment.id))

    return render_template('create_assignment.html')

import os

def create_new_folder(folder_path):
    try:
        # Check if the folder already exists
        if not os.path.exists(folder_path):
            # Create the folder
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        else:
            print(f"Folder '{folder_path}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage



@app.route('/view_assignments', methods=['GET'])
def view_assignments():
    assignments = Assignment.query.all()
    return render_template('view_assignments.html', assignments=assignments)


@app.route('/add_questions/<int:assignment_id>', methods=['GET', 'POST'])
def add_questions(assignment_id):
    if 'teacher_id' not in session:
        flash('You must be logged in to submit your work.')
        return redirect(url_for('teacher_login'))
    assignment = Assignment.query.get_or_404(assignment_id)
    questions = Question.query.filter_by(ass_id=assignment_id).all()
    # Calculate the number of questions and total marks
    total_question_marks = float(sum([question.marks for question in questions]))

    
    if len(questions) >= assignment.num_questions:
        flash(f'You cannot add more than {assignment.num_questions} questions.', 'danger')
        return redirect(url_for('view_assignment', assignment_id=assignment_id))

    if request.method == 'POST':
        question_text = request.form['question']
        marks = float(request.form['marks'])
        question_type = request.form['type']
        optional = False
        if question_type == 'Optional':
            optional = True
        # Check if the total marks exceed the allowed limit
        if total_question_marks + marks > assignment.total_marks:
            flash('The total marks of the questions exceed the assignment\'s allowed total marks.', 'danger')
            return redirect(url_for('view_assignment', assignment_id=assignment_id))

        # Create a Question object
        new_question = Question(
            ass_id=assignment_id,
            question=question_text,
            marks=marks,
            optional=optional
        )

        db.session.add(new_question)
        db.session.commit()

        flash('Question added successfully!', 'success')
        return redirect(url_for('add_questions', assignment_id=assignment_id))

    return render_template('add_questions.html', assignment=assignment, questions=questions)

@app.route('/add_test_cases/<int:question_id>', methods=['GET', 'POST'])
def add_test_cases(question_id):
    if request.method == 'POST':
        if 'teacher_id' not in session:
            flash('You must be logged in to submit your work.')
            return redirect(url_for('teacher_login'))
        case_input = request.form['case']
        output = request.form['output']

        # Create a Testcase object
        new_testcase = Testcase(
            ques_id=question_id,
            case=case_input,
            output=output
        )

        db.session.add(new_testcase)
        db.session.commit()

        flash('Test case added successfully!', 'success')
        return redirect(url_for('add_test_cases', question_id=question_id))

    question = Question.query.get_or_404(question_id)
    testcases = Testcase.query.filter_by(ques_id=question_id).all()

    return render_template('add_test_cases.html', question=question, testcases=testcases)

@app.route('/delete_testcase/<int:testcase_id>', methods=['POST'])
def delete_testcase(testcase_id):
    if 'teacher_id' not in session:
        flash('You must be logged in to submit your work.')
        return redirect(url_for('teacher_login'))
    testcase = Testcase.query.get_or_404(testcase_id)

    db.session.delete(testcase)
    db.session.commit()

    flash('Test case deleted successfully!', 'success')
    return redirect(url_for('edit_question', question_id=testcase.ques_id))


@app.route('/edit_question/<int:question_id>', methods=['GET', 'POST'])
def edit_question(question_id):
    question = Question.query.get_or_404(question_id)

    if request.method == 'POST':
        if 'teacher_id' not in session:
            flash('You must be logged in to submit your work.')
            return redirect(url_for('teacher_login'))
        # Update question details
        question.question = request.form['question']
        question.marks = request.form['marks']
        db.session.commit()

        # Add a new test case if provided
        new_case = request.form.get('case')
        new_output = request.form.get('output')
        print("New case:", new_case)
        print("New output:", new_output)
        if new_case and new_output:
            new_testcase = Testcase(
                ques_id=question.id,
                case=new_case,
                output=new_output
            )
            db.session.add(new_testcase)
            db.session.commit()
            flash('New test case added!', 'success')

        flash('Question updated successfully!', 'success')
        return redirect(url_for('edit_question', question_id=question_id))

    # Retrieve all existing test cases for the question
    testcases = Testcase.query.filter_by(ques_id=question_id).all()

    return render_template('edit_question.html', question=question, testcases=testcases)


@app.route('/edit_testcase/<int:testcase_id>', methods=['GET', 'POST'])
def edit_testcase(testcase_id):
    testcase = Testcase.query.get_or_404(testcase_id)

    if request.method == 'POST':
        if 'teacher_id' not in session:
            flash('You must be logged in to submit your work.')
            return redirect(url_for('teacher_login'))
        testcase.case = request.form['case']
        testcase.output = request.form['output']

        db.session.commit()
        flash('Test case updated successfully!', 'success')
        return redirect(url_for('view_assignment', assignment_id=testcase.question.ass_id))

    return render_template('edit_testcase.html', testcase=testcase)


@app.route('/view_assignment/<int:assignment_id>', methods=['GET'])
def view_assignment(assignment_id):
    assignment = Assignment.query.get_or_404(assignment_id)
    questions = Question.query.filter_by(ass_id=assignment_id).all()

    question_details = []
    total_question_marks = 0
    for question in questions:
        testcases = Testcase.query.filter_by(ques_id=question.id).all()
        question_details.append({
            'question': question,
            'testcases': testcases,
        })
        total_question_marks += question.marks
    return render_template('view_assignment.html',
                           assignment=assignment,
                           student_id=None,
                           question_details=question_details,
                           total_marks=total_question_marks,
                           submission_details=None)

@app.route('/view_assignment_student/<int:assignment_id>', methods=['GET'])
@login_required
def view_assignment_student(assignment_id):
    if 'student_id' not in session:
        flash('You must be logged in to submit your work.')
        return redirect(url_for('student_login'))

    current_student_id = session['student_id']
    assignment = Assignment.query.get_or_404(assignment_id)
    questions = Question.query.filter_by(ass_id=assignment_id).all()
    student_data = Student.query.get_or_404(current_student_id)
    student_group = student_data.group
    current_date = datetime.today().date()
    if student_group == 'A1':
        is_due_date_passed = True if current_date > assignment.due_date1 else False
    else:
        is_due_date_passed = True if current_date > assignment.due_date2 else False
    question_details = []
    total_maks_gained = 0
    for question in questions:
        testcases = Testcase.query.filter_by(ques_id=question.id).all()
        testcase_submissions = {}
        for testcase in testcases:
            submission = Submission.query.filter_by(st_id=current_student_id, ques_id=question.id,
                                                    test_case_id=testcase.id).order_by(Submission.id.desc()).first()
            if submission:
                total_maks_gained += submission.marks
                testcase_submissions[testcase.id] = (submission.output, submission.marks)
            else:
                testcase_submissions[testcase.id] = None  # No submission found for this test case

        question_details.append({
            'question': question,
            'testcases': testcases,
            'testcase_submissions': testcase_submissions  # Storing test case outputs
        })

    return render_template('view_assignment_student.html',
                           assignment=assignment,
                           question_details=question_details,
                           student_id=current_student_id,
                           total_marks_gained=total_maks_gained,
                           due_date_passed=is_due_date_passed)

@app.route('/delete_assignment/<int:assignment_id>', methods=['POST'])
def delete_assignment(assignment_id):
    assignment = Assignment.query.get_or_404(assignment_id)

    # Delete the assignment and its associated questions and test cases
    db.session.delete(assignment)
    db.session.commit()

    flash('Assignment deleted successfully!', 'success')
    return redirect(url_for('view_assignments'))

@app.route('/edit_assignment/<int:assignment_id>', methods=['GET', 'POST'])
def edit_assignment(assignment_id):
    assignment = Assignment.query.get_or_404(assignment_id)

    if request.method == 'POST':
        # Get the updated data from the form
        assignment.topic = request.form['topic']
        assignment.total_marks = float(request.form['total_marks'])
        assignment.date1 = request.form['date1']
        assignment.date2 = request.form['date2']
        assignment.due_date1 = request.form['due_date1']
        assignment.due_date2 = request.form['due_date2']
        assignment.num_questions = int(request.form['num_questions'])

        # Save the updated assignment to the database
        db.session.commit()
        flash('Assignment updated successfully!', 'success')
        return redirect(url_for('view_assignment', assignment_id=assignment.id))

    return render_template('edit_assignment.html', assignment=assignment)


@app.route('/upload_submission/<int:question_id>/<int:assignment_id>', methods=['POST'])
@login_required
def upload_submission(question_id, assignment_id):
    if 'student_id' not in session:
        flash('You must be logged in to submit your work.')
        return redirect(url_for('login'))

    current_student_id = session['student_id']
    student = Student.query.filter_by(id=current_student_id).first()
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('view_assignment_student', assignment_id=assignment_id))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('view_assignment_student', assignment_id=assignment_id))

    if file and allowed_file(file.filename):
        # Ensure the filename is .py instead of .c
        filename = secure_filename(f"{student}.py")
        topic = (Assignment.query.filter_by(id=assignment_id).first()).topic
        filepath = os.path.join(topic, filename)
        file.save(filepath)

        # Run the Python file evaluation
        run_process = Popen(['python', filepath], stdout=PIPE, stderr=PIPE, text=True, encoding='utf-8')
        run_output, run_errors = run_process.communicate()

        if run_process.returncode != 0:
            flash(f"Runtime error: {run_errors}", "error")
            return redirect(url_for('view_assignment_student', assignment_id=assignment_id))
        else:
            print("Program Output:", run_output)

        # Now, run the program with test cases
        test_cases = Testcase.query.filter_by(ques_id=question_id).all()
        question_data = Question.query.filter_by(id=question_id).first()
        test_cases_passed = 0
        for idx, test_case_row in enumerate(test_cases):
            print(f"TEST CASE {idx+1}:")
            test_case = test_case_row.case
            if '<>' in test_case:   # No input required
                print("Running program:", filepath)
                run_process = Popen(['python', filepath], stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding='utf-8')
                try:
                    output, errors = run_process.communicate(timeout=10)
                except subprocess.TimeoutExpired as t_err:
                    flash("Code Timeout during runtime", "error")
                    return redirect(url_for('view_assignment_student', assignment_id=assignment_id))
            elif ';' in test_case:  # Contains multiple inputs
                test_case = '\n'.join(test_case.split(';'))
                print("Running program:", filepath, '<', test_case)
                run_process = Popen(['python', filepath], text=True, encoding='utf-8')
                try:
                    output, errors = run_process.communicate(timeout=10, input=test_case)
                except subprocess.TimeoutExpired as t_err:
                    flash("Code Timeout during runtime", "error")
                    return redirect(url_for('view_assignment_student', assignment_id=assignment_id))
            else:   # Contains single input
                print("Running program:", filepath, '<', test_case)
                run_process = Popen(['python', filepath], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True,
                                    encoding='utf-8')
                try:
                    output, errors = run_process.communicate(timeout=10, input=test_case)
                except subprocess.TimeoutExpired as t_err:
                    flash("Code Timeout during runtime", "error")
                    return redirect(url_for('view_assignment_student', assignment_id=assignment_id))
            desired_output = test_case_row.output
            if run_process.returncode != 0:
                flash(f"Runtime error: {errors}", "error")
                return redirect(url_for('view_assignment_student', assignment_id=assignment_id))
            else:   # Code ran without runtime errors
                # output = output.decode('utf-8')
                if desired_output == '<>':  # Any output is acceptable
                    marks = float(question_data.marks)/len(test_cases)
                    # test_cases_passed = -1
                elif ';' in desired_output: # Contains multiple outputs
                    desired_output = '\n'.join(desired_output.split(';'))
                    if output == desired_output:
                        # test_cases_passed += 1
                        marks = float(question_data.marks)/len(test_cases)
                    else:
                        print(f"Code Output - Desired Output Mismatch\nCode output:\t{output}\nDesired output:\t{desired_output}")
                        marks = 0.0
                else:   # Contains single output
                    if output == desired_output:
                        # test_cases_passed += 1
                        marks = float(question_data.marks)/len(test_cases)
                    else:
                        print(
                            f"Code Output - Desired Output Mismatch\nCode output:\t{output}\nDesired output:\t{desired_output}")
                        marks = 0.0
                test_case_id = test_case_row.id

                # Save the marks to the database
                current_date = datetime.today()
                submission = Submission(st_id=current_student_id, date=current_date,
                                        ass_id=assignment_id, ques_id=question_id, marks=marks,
                                        test_case_id=test_case_id, output=output)
                db.session.add(submission)
                db.session.commit()

                flash(f"File uploaded and evaluated. You received {marks} marks.")
        return redirect(url_for('view_assignment_student',
                                assignment_id=assignment_id))

    flash('Invalid file format. Only .py files are allowed.', 'error')
    return redirect(url_for('view_assignment_student', assignment_id=assignment_id))

@app.route('/view/<filename>/<string:topic>', methods=['GET'])
def view_file(filename,topic):
    topic =os.path.join(topic, filename)
    file_path=os.path.join(os.getcwd(), topic)
    
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return render_template('view_file.html', filename=filename, content=content)
    except Exception as e:
        return f"Error: {e}", 404


from flask import send_from_directory

@app.route('/download/<filename>/<string:topic>', methods=['GET'])
def download_file(filename,topic):
    file_path=os.path.join(os.getcwd(), topic)
    return send_from_directory(file_path, filename, as_attachment=True)


from flask import jsonify

@app.route('/view_uploads/<int:assignment_id>')
def view_uploads(assignment_id):
    # Query the database to get the topic for the given assignment_id
    assignment = Assignment.query.filter_by(id=assignment_id).first()
    topic = assignment.topic  # Extract topic from the queried assignment

    # Construct the folder path based on the topic
    fetch_folder = os.path.join(os.getcwd(), topic)

    # Get the list of files in the folder
    uploads = list_files_with_details(fetch_folder)
    
    # Pass both 'uploads' and 'topic' to the template
    return render_template('view_uploads.html', uploads=uploads, topic=topic)


import os

def list_files_with_details(folder_path):
    """
    Returns a list of dictionaries with file names and their full paths in the specified folder.

    Parameters:
        folder_path (str): The path to the folder to scan for files.

    Returns:
        list: A list of dictionaries, where each dictionary has 'file_name' and 'file_path'.
    """
    files_with_details = []
    
    try:
        # Iterate through the folder's content
        for entry in os.scandir(folder_path):
            if entry.is_file():  # Check if it's a file
                files_with_details.append({
                    'file_name': entry.name,  # File name only
                    'file_path': entry.path  # Full file path
                })
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return files_with_details

# Example usage




@app.route('/run_code/<int:question_id>/<int:assignment_id>', methods=['POST'])
@login_required
def run_code(question_id, assignment_id):
    current_student_id = session['student_id']
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('view_assignment_student', assignment_id=assignment_id))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('view_assignment_student', assignment_id=assignment_id))

    if file and allowed_file(file.filename):
        # Ensure the file is a Python file (.py)
        if not file.filename.endswith('.py'):
            flash('Please upload a valid Python file (.py)', 'error')
            return redirect(url_for('view_assignment_student', assignment_id=assignment_id))

        filename = secure_filename(f"{assignment_id}_{current_student_id}_{question_id}.py")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Now, run the Python file with test cases
        test_cases = Testcase.query.filter_by(ques_id=question_id).all()
        question_data = Question.query.filter_by(id=question_id).first()
        submissions = []
        for idx, test_case_row in enumerate(test_cases):
            submission_data = dict()
            print(f"TEST CASE {idx + 1}:")
            test_case = test_case_row.case
            if '<>' in test_case:  # No input required
                print("Running Python program:", filepath)
                run_process = Popen(['python3', filepath], stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding='utf-8')
                try:
                    output, errors = run_process.communicate(timeout=10)
                except subprocess.TimeoutExpired as t_err:
                    flash("Code Timeout during runtime", 'error')
                    return redirect(url_for('view_assignment_student', assignment_id=assignment_id))
            elif ';' in test_case:  # Contains multiple inputs
                test_case = '\n'.join(test_case.split(';'))
                print("Running Python program:", filepath, '<', test_case)
                run_process = Popen(['python3', filepath], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True, encoding='utf-8')
                try:
                    output, errors = run_process.communicate(timeout=10, input=test_case)
                except subprocess.TimeoutExpired as t_err:
                    flash("Code Timeout during runtime", "error")
                    return redirect(url_for('view_assignment_student', assignment_id=assignment_id))
            else:  # Contains single input
                print("Running Python program:", filepath, '<', test_case)
                run_process = Popen(['python3', filepath], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True, encoding='utf-8')
                try:
                    output, errors = run_process.communicate(timeout=10, input=test_case)
                except subprocess.TimeoutExpired as t_err:
                    flash("Code Timeout during runtime", "error")
                    return redirect(url_for('view_assignment_student', assignment_id=assignment_id))

            desired_output = test_case_row.output
            submission_data['case'] = test_case
            submission_data['output'] = desired_output
            submission_data['st_output'] = output
            if run_process.returncode != 0:
                flash(f"Runtime error: {errors}", "error")
                return redirect(url_for('view_assignment_student', assignment_id=assignment_id))
            else:  # Code ran without runtime errors
                output = output
                print("Run output:", output)
                if desired_output == '<>':  # Any output is acceptable
                    submission_data['status'] = "Correct"
                elif ';' in desired_output:  # Contains multiple outputs
                    desired_output = '\n'.join(desired_output.split(';'))
                    if output == desired_output:
                        submission_data['status'] = "Correct"
                    else:
                        print(f"Code Output - Desired Output Mismatch\nCode output:\t{output}\nDesired output:\t{desired_output}")
                        submission_data['status'] = "Incorrect"
                else:  # Contains single output
                    try:
                        output = float(output)
                        desired_output = float(desired_output)
                        if output == desired_output:
                            submission_data['status'] = "Correct"
                        else:
                            print(f"Code Output - Desired Output Mismatch\nCode output:\t{output}\nDesired output:\t{desired_output}")
                            submission_data['status'] = "Incorrect"
                    except Exception as err:
                        if output == desired_output:
                            submission_data['status'] = "Correct"
                        else:
                            print(f"Code Output - Desired Output Mismatch\nCode output:\t{output}\nDesired output:\t{desired_output}")
                            submission_data['status'] = "Incorrect"
                submissions.append(submission_data)

        return render_template('run_code.html',
                               assignment_id=assignment_id,
                               student_id=current_student_id,
                               question=question_data.question,
                               submissions=submissions
                               )



@app.route('/view_all_student_marks', methods=['GET'])
def view_all_student_marks():
    students = Student.query.all()
    assignments = Assignment.query.all()

    
    student_marks = {}
    for student in students:
        student_data = {
            'name': student.name,
            'roll': student.roll,
            'group': student.group,
            'assignments': []
        }

        for assignment in assignments:
            
            total_marks = db.session.query(func.sum(Submission.marks)) \
                .filter(
                Submission.st_id == student.id,
                Submission.ass_id == assignment.id
            ).scalar()

            
            total_marks = total_marks if total_marks is not None else 0

            assignment_info = {
                'topic': assignment.topic,
                'date': assignment.date1 if student.group == 'A1' else assignment.date2,
                'marks': total_marks
            }
            student_data['assignments'].append(assignment_info)

        student_marks[student.id] = student_data

    return render_template(
        'view_all_student_marks.html',
        student_marks=student_marks,
        assignments=assignments
    )


@app.route('/download_group_csv/<int:group>')
def download_group_csv(group):
    
    group_name = 'A1' if group == 1 else 'A2'
    date_field = 'date1' if group == 1 else 'date2'

    
    students = Student.query.filter_by(group=group_name).all()
    assignments = Assignment.query.all()

    
    csv_data = StringIO()
    writer = csv.writer(csv_data)

    
    writer.writerow(
        ['Sl. No.', 'Student Name', 'Student Roll'] + [f"Day {i + 1} ({a.topic}, {getattr(a, date_field)})" for i, a in
                                                       enumerate(assignments)])

    
    for index, student in enumerate(students, start=1):
        row = [index, student.name, student.roll]
        for assignment in assignments:
            submission = Submission.query.filter_by(st_id=student.id, ass_id=assignment.id).all()
            total_marks = sum(sub.marks for sub in submission) if submission else 0
            row.append(total_marks)
        writer.writerow(row)

    
    csv_data.seek(0)
    filename = f"Group_{group_name}_Marks.csv"
    return Response(csv_data, mimetype='text/csv', headers={"Content-Disposition": f"attachment;filename={filename}"})

from flask import Flask, send_from_directory
import os


@app.route('/custom_static/<filename>')
def custom_static(filename):
   # Ensure the path to your report directory is correct
   # report_dir = Path(r'C:\Users\Rhythm\OneDrive\Desktop\Online_Programming_Assignment_Portal-main (2)\Online_Programming_Assignment_Portal-main\Online_Programming_Assignment_Portal-main\src\report')
    return send_from_directory("report", filename)

from flask import send_file



import os
from flask import send_file, abort

@app.route('/view_pdf')
def view_pdf():
    pdf_path = r"C:\Users\Rhythm\Downloads\date 6 latest frontend\06-03-2025\Online_Programming_Assignment_Portal-main\src\report\similarity_report.pdf"  # Update the correct path

    if not os.path.exists(pdf_path):
        return abort(404, description="PDF file not found.")
    os.chmod(pdf_path, 0o644)

    return send_file(pdf_path, mimetype="application/pdf", as_attachment=False)


@app.route('/check_similarity/<int:assignment_id>')
def check_similarity(assignment_id):
    topic = (Assignment.query.filter_by(id=assignment_id).first()).topic
    if topic is None:
        return "Error: No topic found for the given assignment ID.", 400

    start_folder = os.getcwd()
    folder_path = find_subfolder_path(start_folder, topic)

    if folder_path is None:
        return f"Error: The folder '{topic}' was not found in '{start_folder}'.", 400

    output_dir = os.path.join(os.getcwd(), "report")
    output_dir2 = os.path.join(os.getcwd(), "templates")

    codes = load_codes_from_folder(folder_path)
    similarities = calculate_pairwise_similarities(codes)
    clusters, similarity_matrix, files = cluster_codes(similarities, codes)

    print("Code Clusters based on Approach:")
    for cluster_id, cluster_files in clusters.items():
        print(f"Cluster {cluster_id}: {cluster_files}" if cluster_id != -1 else f"Outliers: {cluster_files}")

    generate_report(clusters, similarities, codes, similarity_matrix, files, output_dir, output_dir2)
    
    return render_template('similarity_options.html', assignment_id=assignment_id)


def find_subfolder_path(start_folder, subfolder_name):
    for root, dirs, files in os.walk(start_folder):
        if subfolder_name in dirs:
            return os.path.join(root, subfolder_name)
    return None

@app.route('/show_clusters')
def show_clusters():
    """Route to display cluster data visualization."""
    
    return render_template('similarity_graph.html')

@app.route('/show_heatmap')
def show_heatmap():
    
    
    return render_template('heatmap.html')

@app.route('/show_report')
def show_report():
    
    
    return render_template('report.html')
