from pydantic import BaseModel

class ProfileUpdate(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: str
    specialty: str
    institution: str
    hospital_name: str = ""

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

class NotificationPrefs(BaseModel):
    critical_alerts: bool
    report_ready: bool
    weekly_digest: bool
    model_updates: bool

class ThemeUpdate(BaseModel):
    dark_mode: bool
