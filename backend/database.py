from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, ForeignKey, Date, Time, inspect, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from .config import DATABASE_URL
from datetime import datetime
Base = declarative_base()

class Doctor(Base):
    __tablename__ = 'doctors'
    doctor_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    specialization = Column(String, nullable=False)
    phone = Column(String)
    # email = Column(String)
    working_days = relationship("DoctorWorkingDay", back_populates="doctor")
    appointments = relationship("Appointment", back_populates="doctor")

class DoctorWorkingDay(Base):
    __tablename__ = 'doctor_working_days'
    working_day_id = Column(Integer, primary_key=True)
    doctor_id = Column(Integer, ForeignKey('doctors.doctor_id'))
    day_of_week = Column(String, nullable=False) 
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    doctor = relationship("Doctor", back_populates="working_days")

class Patient(Base):
    __tablename__ = 'patients'
    patient_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    email = Column(String)
    date_of_birth = Column(Date)
    address = Column(String)
    appointments = relationship("Appointment", back_populates="patient")
    medications = relationship("Medication", back_populates="patient")
    medical_conditions = relationship("MedicalCondition", back_populates="patient")

    def __repr__(self):
        return f"<Patient(id={self.patient_id}, name={self.name})>"

class Appointment(Base):
    __tablename__ = 'appointments'
    appointment_id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patients.patient_id'))
    doctor_id = Column(Integer, ForeignKey('doctors.doctor_id'))
    datetime = Column(DateTime, nullable=False)
    appointment_type = Column(String)  # Add this line
    notes = Column(String)  # Add this line
    # duration = Column(Integer, default=30)  # appointment duration in minutes
    status = Column(String, default='scheduled')  # scheduled, confirmed, cancelled, completed
    patient = relationship("Patient", back_populates="appointments")
    doctor = relationship("Doctor", back_populates="appointments")
    

class Medication(Base):
    __tablename__ = 'medications'
    medication_id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.patient_id'))
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    dosage = Column(String)
    frequency = Column(String)
    start_date = Column(Date)
    end_date = Column(Date)
    patient = relationship("Patient", back_populates="medications")

class MedicalCondition(Base):
    """Medical condition model."""
    __tablename__ = 'medical_conditions'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.patient_id'), nullable=False)
    condition_name = Column(String(255), nullable=False)
    date_recorded = Column(DateTime, default=datetime.now)
    notes = Column(Text, nullable=True)
    
    # Relationships
    patient = relationship("Patient", back_populates="medical_conditions")
    symptoms = relationship("Symptom", back_populates="condition", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<MedicalCondition(id={self.id}, patient_id={self.patient_id}, condition={self.condition_name})>"

class Symptom(Base):
    """Symptom model."""
    __tablename__ = 'symptoms'
    
    id = Column(Integer, primary_key=True)
    condition_id = Column(Integer, ForeignKey('medical_conditions.id'), nullable=False)
    symptom_name = Column(String(255), nullable=False)
    severity = Column(String(50), nullable=True)
    
    # Relationships
    condition = relationship("MedicalCondition", back_populates="symptoms")
    
    def __repr__(self):
        return f"<Symptom(id={self.id}, condition_id={self.condition_id}, name={self.symptom_name})>"

# Initialize database
engine = create_engine(DATABASE_URL)
inspector = inspect(engine)
# Only create tables if they don't exist
if not inspector.has_table("doctors"):  # Check if at least one main table exists
    Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def init_db():
    """Initialize the database."""
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    # Only create tables if they don't exist
    if not inspector.has_table("doctors"):  # Check if at least one main table exists
        Base.metadata.create_all(engine)
    return engine

def get_session():
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()
