import os, pytest, pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from app.database import Base, get_db
from app.main import app
from app.config import settings
from app.utils.security import hash_password
from app.utils.rate_limit import clear_rate_limit

TEST_DB_URL = "sqlite+aiosqlite://"

@pytest.fixture(scope="session")
def session_engine():
    return create_async_engine(TEST_DB_URL, echo=False)

@pytest.fixture(autouse=True)
def _clear_rate_limits():
    clear_rate_limit("login:127.0.0.1")
    yield
    clear_rate_limit("login:127.0.0.1")


@pytest_asyncio.fixture(autouse=True)
async def setup_db(session_engine):
    async with session_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with session_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture
async def db_session(session_engine):
    factory = async_sessionmaker(session_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

@pytest_asyncio.fixture
async def client(db_session):
    async def _get_db():
        yield db_session
    app.dependency_overrides[get_db] = _get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()

@pytest_asyncio.fixture
async def seed_user(db_session):
    from app.models.user import User
    user = User(
        email="dr.rajan@apollo.org",
        password_hash=hash_password("demo1234"),
        first_name="Anil", last_name="Rajan",
        role="Senior Ophthalmologist",
    )
    db_session.add(user)
    await db_session.commit()
    return user

@pytest_asyncio.fixture
async def auth_headers(client, seed_user):
    r = await client.post("/api/auth/login", json={"email": "dr.rajan@apollo.org", "password": "demo1234"})
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}"}
