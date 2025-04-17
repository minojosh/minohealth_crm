import sqlite3
from datetime import datetime, timedelta, date
import random

def create_database():
    conn = sqlite3.connect('backend/healthcare.db')
    cursor = conn.cursor()
    
    # Drop existing tables if they exist
    tables = [
        'doctor_working_days',
        'interaction_logs',
        'patient_medications',
        'medications',
        'appointments',
        'patients',
        'doctors'
    ]
    
    for table in tables:
        cursor.execute(f'DROP TABLE IF EXISTS {table}')
    
    # Create Patients table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        phone TEXT NOT NULL,
        email TEXT,
        date_of_birth DATE,
        address TEXT
    )
    ''')
    
    # Create Doctors table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS doctors (
        doctor_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        specialization TEXT,
        phone TEXT
    )
    ''')
    
    # Create Appointments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS appointments (
        appointment_id INTEGER PRIMARY KEY,
        patient_id INTEGER,
        doctor_id INTEGER,
        datetime DATETIME NOT NULL,
        status TEXT DEFAULT 'scheduled',
        appointment_type TEXT,
        notes TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id),
        FOREIGN KEY (doctor_id) REFERENCES doctors (doctor_id)
    )
    ''')
    
    # Create Medications table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS medications (          
        medication_id INTEGER PRIMARY KEY,
        patient_id INTEGER,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        dosage TEXT,
        frequency TEXT,
        start_date DATE NOT NULL,
        end_date DATE NOT NULL       
    )
    ''')
    
    # Create Patient_Medications table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patient_medications (
        prescription_id INTEGER PRIMARY KEY,
        patient_id INTEGER,
        medication_id INTEGER,
        prescribed_date DATE,
        end_date DATE,
        dosage TEXT,
        frequency TEXT,
        refills_remaining INTEGER,
        last_refill_date DATE,
        next_refill_date DATE,
        status TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id),
        FOREIGN KEY (medication_id) REFERENCES medications (medication_id)
    )
    ''')
    
    # Create Interaction_Logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interaction_logs (
        log_id INTEGER PRIMARY KEY,
        patient_id INTEGER,
        interaction_type TEXT,
        interaction_date DATETIME,
        notes TEXT,
        response_recorded TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
    )
    ''')

    # Create doctor_working_days table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS doctor_working_days (
        working_day_id INTEGER PRIMARY KEY,
        doctor_id INTEGER,
        day_of_week TEXT NOT NULL,
        start_time TIME NOT NULL,
        end_time TIME NOT NULL,
        FOREIGN KEY (doctor_id) REFERENCES doctors (doctor_id)
    )
    ''')
    
    conn.commit()
    return conn

def insert_test_data(conn):
    cursor = conn.cursor()
    
    # Insert test patients
    patients = [
        ('Emma Thompson', '0501234567', 'emma.thompson@email.com', '1985-03-15', '123 Maple Street'),
        ('David Rodriguez', '0507654321', 'david.rodriguez@email.com', '1978-11-22', '456 Oak Avenue'),
        ('Sophia Kim', '0509876543', 'sophia.kim@email.com', '1990-07-10', '789 Pine Road'),
        ('Marcus Johnson', '0502345678', 'marcus.johnson@email.com', '1965-12-05', '321 Cedar Lane'),
        ('Aisha Patel', '0506789012', 'aisha.patel@email.com', '1995-09-18', '654 Birch Street')
    ]
    try:
        print("Inserting patients...")
        cursor.executemany('INSERT INTO patients (name, phone, email, date_of_birth, address) VALUES (?, ?, ?, ?, ?)', patients)
        conn.commit()
        print(f"Inserted {len(patients)} patients successfully")
    except sqlite3.Error as e:
        print(f"Error inserting patients: {e}")
        conn.rollback()
        raise
    
    # Insert test doctors
    doctors = [
        ('Dr. Elena Martinez', 'Cardiology', '0551112222'),
        ('Dr. Alex Chen', 'Neurology', '0553334444'),
        ('Dr. Rachel Kumar', 'Pediatrics', '0555556666'),
        ('Dr. Michael Thompson', 'General Practice', '0557778888')
    ]
    try:
        print("Inserting doctors...")
        cursor.executemany('INSERT INTO doctors (name, specialization, phone) VALUES (?, ?, ?)', doctors)
        conn.commit()
        print(f"Inserted {len(doctors)} doctors successfully")
    except sqlite3.Error as e:
        print(f"Error inserting doctors: {e}")
        conn.rollback()
        raise
    
    # Insert test medications

    medications = [
        (1, 'Atorvastatin', 'Cholesterol medication', '40mg', 'Once daily', 
        date.today(), date.today() + timedelta(days=30)),
        (2, 'Metoprolol', 'Blood pressure medication', '50mg', 'Twice daily', 
        date.today(), date.today() + timedelta(days=30)),
        (3, 'Levothyroxine', 'Thyroid hormone', '75mcg', 'Once daily',
        date.today(), date.today() + timedelta(days=30)),
        (4, 'Pantoprazole', 'Acid reflux treatment', '40mg', 'Once daily',
        date.today(), date.today() + timedelta(days=30)),
        (5, 'Fluoxetine', 'Antidepressant', '20mg', 'Once daily',
        date.today(), date.today() + timedelta(days=30))
    ]
    try:
        print("Inserting medications...")
        cursor.executemany('''
        INSERT INTO medications 
        (patient_id, name, description, dosage, frequency, start_date, end_date) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', medications)
        conn.commit()
        print(f"Inserted {len(medications)} medications successfully")
    except sqlite3.Error as e:
        print(f"Error inserting medications: {e}")
        conn.rollback()
        raise
    
    # Insert test appointments
    base_date = datetime.now()
    appointments = []
    for i in range(1, 6):  
        for j in range(2):  #
            appointment_date = base_date + timedelta(days=random.randint(24, 25))
            hour = random.randint(9, 16)  
            minute = random.choice([0, 15, 30, 45])
            
            appointment_time = appointment_date.replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0
            )
            
            appointments.append((
                i,  # patient_id
                random.randint(1, 4),  # doctor_id
                appointment_time.strftime('%Y-%m-%d %H:%M:%S'),
                'scheduled',
                random.choice(['Regular Checkup', 'Follow-up', 'Annual Physical']),
                'Patient consultation notes'
            ))
    try:
        print("Inserting appointments...")
        cursor.executemany('INSERT INTO appointments (patient_id, doctor_id, datetime, status, appointment_type, notes) VALUES (?, ?, ?, ?, ?, ?)', appointments)
        conn.commit()
        print(f"Inserted {len(appointments)} appointments successfully")
    except sqlite3.Error as e:
        print(f"Error inserting appointments: {e}")
        conn.rollback()
        raise
    
    # Insert test patient medications
    current_date = datetime.now().date()
    prescriptions = []
    for i in range(1, 6): 
        for j in range(1, 3):  
            prescribed_date = current_date - timedelta(days=random.randint(1, 60))
            end_date = prescribed_date + timedelta(days=90)
            next_refill = prescribed_date + timedelta(days=30)
            prescriptions.append((
                i,  # patient_id
                random.randint(1, 5),  # medication_id
                prescribed_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                'As prescribed',
                'Daily',
                3,
                prescribed_date.strftime('%Y-%m-%d'),
                next_refill.strftime('%Y-%m-%d'),
                'active'
            ))
    try:
        print("Inserting patient medications...")
        cursor.executemany('''INSERT INTO patient_medications 
                             (patient_id, medication_id, prescribed_date, end_date, dosage, frequency, 
                              refills_remaining, last_refill_date, next_refill_date, status) 
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', prescriptions)
        conn.commit()
        print(f"Inserted {len(prescriptions)} patient medications successfully")
    except sqlite3.Error as e:
        print(f"Error inserting patient medications: {e}")
        conn.rollback()
        raise
    
    # Insert interaction logs
    interaction_logs = []
    for i in range(1, 6):  # For each patient
        interaction_logs.append((
            i,  # patient_id
            random.choice(['Phone', 'Email', 'In-person']),
            (base_date - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d %H:%M:%S'),
            'Routine health inquiry',
            'Patient consulted about general health'
        ))
    try:
        print("Inserting interaction logs...")
        cursor.executemany('INSERT INTO interaction_logs (patient_id, interaction_type, interaction_date, notes, response_recorded) VALUES (?, ?, ?, ?, ?)', interaction_logs)
        conn.commit()
        print(f"Inserted {len(interaction_logs)} interaction logs successfully")
    except sqlite3.Error as e:
        print(f"Error inserting interaction logs: {e}")
        conn.rollback()
        raise
    
    # Insert test working days for doctors
    cursor.execute('DELETE FROM doctor_working_days')  # Clear existing data
    working_days = [
        # For Dr. Elena Martinez (ID: 1)
        (1, 'Monday', '09:00', '17:00'),
        (1, 'Wednesday', '09:00', '17:00'),
        (1, 'Friday', '09:00', '17:00'),
        # For Dr. Alex Chen (ID: 2)
        (2, 'Tuesday', '08:00', '16:00'),
        (2, 'Thursday', '08:00', '16:00'),
        # For Dr. Rachel Kumar (ID: 3)
        (3, 'Monday', '10:00', '18:00'),
        (3, 'Thursday', '10:00', '18:00'),
        # For Dr. Michael Thompson (ID: 4)
        (4, 'Wednesday', '09:00', '17:00'),
        (4, 'Friday', '09:00', '17:00')
    ]
    try:
        print("Inserting doctor working days...")
        cursor.executemany('INSERT INTO doctor_working_days (doctor_id, day_of_week, start_time, end_time) VALUES (?, ?, ?, ?)', working_days)
        conn.commit()
        print(f"Inserted {len(working_days)} doctor working days successfully")
    except sqlite3.Error as e:
        print(f"Error inserting doctor working days: {e}")
        conn.rollback()
        raise
    
    conn.commit()

def display_test_data(conn):
    cursor = conn.cursor()
    
    print("\n=== Patients ===")
    cursor.execute('SELECT * FROM patients')
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    
    print("\n=== Doctors ===")
    cursor.execute('SELECT * FROM doctors')
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    
    print("\n=== Appointments ===")
    cursor.execute('SELECT * FROM appointments')
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    
    print("\n=== Medications ===")
    cursor.execute('SELECT * FROM medications')
    rows = cursor.fetchall()
    for row in rows:
        print(row)

def display_doctor_working_days(conn):
    cursor = conn.cursor()
    print("\n=== Doctor Working Days ===")
    cursor.execute('''
        SELECT d.name, dw.day_of_week, dw.start_time, dw.end_time
        FROM doctor_working_days dw
        JOIN doctors d ON d.doctor_id = dw.doctor_id
        ORDER BY d.name, dw.day_of_week
    ''')
    rows = cursor.fetchall()
    for row in rows:
        print(f"Doctor: {row[0]}, Day: {row[1]}, Hours: {row[2]} - {row[3]}")

if __name__ == "__main__":
    try:
        conn = create_database()
        insert_test_data(conn)
        display_test_data(conn)
        display_doctor_working_days(conn)
    finally:
        if 'conn' in locals():
            conn.close()
            print('Database connection closed successfully')
    print('Setup completed successfully')