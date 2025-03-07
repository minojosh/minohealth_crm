from datetime import datetime, timedelta
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_
#from config import DATABASE_URL
from .config import DATABASE_URL
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
#from database import Doctor, DoctorWorkingDay, Appointment, Patient
from .database import Doctor, DoctorWorkingDay, Appointment, Patient

def schedule_appointment(patient_id: int, appointment_datetime: datetime) -> dict:
    """
    Update the appointment datetime for a given patient.
    
    Args:
        patient_id: The ID of the patient
        appointment_datetime: The new datetime for the appointment
        
    Returns:
        dict: A dictionary containing the status and message of the operation
    """
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check if patient exists
        patient = session.query(Patient).filter_by(patient_id=patient_id).first()
        if not patient:
            return {
                "success": False,
                "message": f"Patient with ID {patient_id} not found"
            }
        
        # Find the appointment for the patient
        appointment = session.query(Appointment).filter_by(patient_id=patient_id).first()
        if not appointment:
            return {
                "success": False,
                "message": f"No appointment found for patient ID {patient_id}"
            }
        
        # Update the appointment datetime
        old_datetime = appointment.datetime
        appointment.datetime = appointment_datetime
        
        # Commit the changes
        session.commit()
        
        return {
            "success": True,
            "message": "Appointment updated successfully",
            "data": {
                "patient_id": patient_id,
                "old_datetime": old_datetime,
                "new_datetime": appointment_datetime
            }
        }
        
    except SQLAlchemyError as e:
        session.rollback()
        return {
            "success": False,
            "message": f"Database error: {str(e)}"
        }
    finally:
        session.close()

if __name__ == "__main__":
    # Example usage
    tomorrow = datetime.now() + timedelta(days=1)
    appointment_datetime = tomorrow.replace(
        hour=14, 
        minute=30, 
        second=0, 
        microsecond=0
    )
    
    result = schedule_appointment(
        patient_id=1,
        appointment_datetime=appointment_datetime
    )
    print(result)