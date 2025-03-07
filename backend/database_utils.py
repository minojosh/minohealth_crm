from datetime import datetime
from typing import Dict, Optional, Any
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.session import Session
from .database import Patient, Appointment, Doctor, get_session
import logging

logger = logging.getLogger(__name__)

class PatientDatabaseManager:
    """Manages patient database operations"""
    
    FIELD_MAPPINGS = {
        'name': 'name',
        'phone': 'phone',
        'email': 'email',
        'dob': 'date_of_birth',  # Map YAML 'dob' to database 'date_of_birth'
        'address': 'address'
    }

    def __init__(self):
        self.session = get_session()

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime.date]:
        """Parse date string into datetime.date object."""
        if not date_str:
            return None
            
        formats = ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%d/%m/%Y', '%m/%d/%Y']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        return None

    def create_patient(self, patient_data: Dict[str, Any]) -> Optional[int]:
        """Create a new patient record."""
        try:
            # Convert date of birth if provided
            if date_str := patient_data.get('date_of_birth'):
                patient_data['date_of_birth'] = self._parse_date(date_str)

            # Create new patient
            new_patient = Patient(**patient_data)
            self.session.add(new_patient)
            self.session.commit()
            
            logger.info(f"Created patient: {new_patient.name} (ID: {new_patient.patient_id})")
            return new_patient.patient_id
            
        except SQLAlchemyError as e:
            logger.error(f"Database error creating patient: {str(e)}")
            self.session.rollback()
            return None
        except Exception as e:
            logger.error(f"Error creating patient: {str(e)}")
            self.session.rollback()
            return None
        finally:
            self.session.close()

    def create_appointment(self, patient_id: int, appointment_type: str,
                         scheduled_date: str, scheduled_time: str,
                         doctor_name: Optional[str] = None) -> Optional[int]:
        """Create appointment for a patient."""
        try:
            # Validate inputs
            if not all([patient_id, scheduled_date, scheduled_time]):
                logger.error("Missing required appointment fields")
                return None

            # Parse appointment datetime
            try:
                appointment_datetime = datetime.strptime(
                    f"{scheduled_date} {scheduled_time}",
                    "%Y-%m-%d %H:%M"
                )
            except ValueError as e:
                logger.error(f"Invalid date/time format: {str(e)}")
                return None

            # Find doctor if provided
            doctor_id = None
            if doctor_name:
                doctor = self.session.query(Doctor).filter(
                    Doctor.name.ilike(f"%{doctor_name}%")
                ).first()
                if doctor:
                    doctor_id = doctor.doctor_id
                    logger.info(f"Found doctor: {doctor.name} (ID: {doctor.doctor_id})")
                else:
                    logger.warning(f"Doctor not found: {doctor_name}")

            # Create appointment
            appointment = Appointment(
                patient_id=patient_id,
                doctor_id=doctor_id,
                datetime=appointment_datetime,
                appointment_type=appointment_type,
                status='scheduled'
            )
            
            self.session.add(appointment)
            self.session.commit()
            
            logger.info(f"Created appointment for patient {patient_id}")
            return appointment.appointment_id
            
        except SQLAlchemyError as e:
            logger.error(f"Database error creating appointment: {str(e)}")
            self.session.rollback()
            return None
        finally:
            self.session.close()

    def update_patient(self, patient_id: int, data: Dict[str, Any]) -> bool:
        """Update patient information from extracted data."""
        try:
            patient = self.session.query(Patient).get(patient_id)
            if not patient:
                logger.error(f"Patient not found: ID {patient_id}")
                return False

            updated = False
            for yaml_field, db_field in self.FIELD_MAPPINGS.items():
                if value := data.get(yaml_field):
                    if db_field == 'date_of_birth':
                        value = self._parse_date(value)
                    if value is not None and getattr(patient, db_field) != value:
                        setattr(patient, db_field, value)
                        updated = True
                        logger.info(f"Updated {db_field} for patient {patient_id}")

            if updated:
                self.session.commit()
                logger.info(f"Successfully updated patient {patient_id}")
                return True
            
            logger.info(f"No updates needed for patient {patient_id}")
            return False
            
        except SQLAlchemyError as e:
            logger.error(f"Database error updating patient {patient_id}: {str(e)}")
            self.session.rollback()
            return False
        finally:
            self.session.close()
