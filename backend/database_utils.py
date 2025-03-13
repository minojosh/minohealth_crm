from datetime import datetime
from typing import Dict, Optional, Any, List
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

    def add_medical_condition(self, patient_id: int, condition: Optional[str] = None, symptoms: Optional[List[str]] = None) -> Optional[int]:
        """
        Add medical condition and symptoms for a patient.
        
        Args:
            patient_id: ID of the patient
            condition: Medical condition name
            symptoms: List of symptoms
            
        Returns:
            ID of the created medical condition record or None if failed
        """
        try:
            with get_session() as session:
                # Check if patient exists
                patient = session.query(Patient).filter(Patient.patient_id == patient_id).first()
                if not patient:
                    logger.error(f"Patient with ID {patient_id} not found")
                    return None
                
                # Create medical condition record
                from .database import MedicalCondition, Symptom
                
                # Create condition record
                medical_condition = MedicalCondition(
                    patient_id=patient_id,
                    condition_name=condition if condition else "Unknown",
                    date_recorded=datetime.now()
                )
                session.add(medical_condition)
                session.flush()  # Get the ID without committing
                
                # Add symptoms if provided
                if symptoms and isinstance(symptoms, list):
                    for symptom_name in symptoms:
                        if symptom_name and isinstance(symptom_name, str):
                            symptom = Symptom(
                                condition_id=medical_condition.id,
                                symptom_name=symptom_name
                            )
                            session.add(symptom)
                
                session.commit()
                logger.info(f"Added medical condition {medical_condition.id} for patient {patient_id}")
                return medical_condition.id
                
        except Exception as e:
            logger.error(f"Error adding medical condition: {str(e)}")
            return None

    def get_patient(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """Get patient details by ID."""
        try:
            with get_session() as session:
                patient = session.query(Patient).filter(Patient.patient_id == patient_id).first()
                if not patient:
                    logger.warning(f"Patient with ID {patient_id} not found")
                    return None
                
                # Convert to dictionary
                patient_dict = {
                    'patient_id': patient.patient_id,
                    'name': patient.name,
                    'phone': patient.phone,
                    'email': patient.email,
                    'address': patient.address,
                    'date_of_birth': patient.date_of_birth.isoformat() if patient.date_of_birth else None
                }
                
                return patient_dict
                
        except Exception as e:
            logger.error(f"Error getting patient: {str(e)}")
            return None
            
    def get_patient_appointments(self, patient_id: int) -> List[Dict[str, Any]]:
        """Get appointments for a patient."""
        try:
            with get_session() as session:
                appointments = session.query(Appointment).filter(
                    Appointment.patient_id == patient_id
                ).all()
                
                # Convert to list of dictionaries
                appointment_list = []
                for appt in appointments:
                    doctor_name = None
                    if appt.doctor_id:
                        doctor = session.query(Doctor).filter(Doctor.doctor_id == appt.doctor_id).first()
                        if doctor:
                            doctor_name = doctor.name
                    
                    appointment_list.append({
                        'appointment_id': appt.appointment_id,
                        'datetime': appt.datetime.isoformat() if appt.datetime else None,
                        'appointment_type': appt.appointment_type,
                        'status': appt.status,
                        'doctor_id': appt.doctor_id,
                        'doctor_name': doctor_name,
                        'notes': appt.notes
                    })
                
                return appointment_list
                
        except Exception as e:
            logger.error(f"Error getting patient appointments: {str(e)}")
            return []
            
    def get_patient_medical_conditions(self, patient_id: int) -> List[Dict[str, Any]]:
        """Get medical conditions for a patient."""
        try:
            from .database import MedicalCondition, Symptom
            
            with get_session() as session:
                conditions = session.query(MedicalCondition).filter(
                    MedicalCondition.patient_id == patient_id
                ).all()
                
                # Convert to list of dictionaries
                condition_list = []
                for condition in conditions:
                    # Get symptoms for this condition
                    symptoms = session.query(Symptom).filter(
                        Symptom.condition_id == condition.id
                    ).all()
                    
                    symptom_list = [
                        {
                            'symptom_id': symptom.id,
                            'name': symptom.symptom_name,
                            'severity': symptom.severity
                        }
                        for symptom in symptoms
                    ]
                    
                    condition_list.append({
                        'condition_id': condition.id,
                        'name': condition.condition_name,
                        'date_recorded': condition.date_recorded.isoformat() if condition.date_recorded else None,
                        'notes': condition.notes,
                        'symptoms': symptom_list
                    })
                
                return condition_list
                
        except Exception as e:
            logger.error(f"Error getting patient medical conditions: {str(e)}")
            return []

    def get_all_patients(self) -> List[Dict[str, Any]]:
        """Get all patients."""
        try:
            with get_session() as session:
                patients = session.query(Patient).all()
                
                # Convert to list of dictionaries
                patient_list = []
                for patient in patients:
                    patient_list.append({
                        'patient_id': patient.patient_id,
                        'name': patient.name,
                        'phone': patient.phone,
                        'email': patient.email,
                        'address': patient.address,
                        'date_of_birth': patient.date_of_birth.isoformat() if patient.date_of_birth else None
                    })
                
                return patient_list
                
        except Exception as e:
            logger.error(f"Error getting all patients: {str(e)}")
            return []
