from pydantic import BaseModel, EmailStr

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthUser(BaseModel):
    name: str
    role: str
    initials: str
    email: str

class LoginResponse(BaseModel):
    token: str
    user: AuthUser

class ForgotPasswordRequest(BaseModel):
    email: str

class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: str
    password: str

class RequestAccessRequest(BaseModel):
    name: str
    email: str
    institution: str
    reason: str = ""
