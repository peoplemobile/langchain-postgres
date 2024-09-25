import contextlib
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from langchain_core.documents import Document
from langchain_core.load import Serializable, dumps, loads
from langchain_core.stores import BaseStore
from sqlalchemy import Engine, String, Text, and_, create_engine, delete, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import (
    Mapped,
    Session,
    declarative_base,
    mapped_column,
    sessionmaker,
)

Base = declarative_base()

V = TypeVar("V")


class LangchainKeyValueStores(Base):  # type: ignore[valid-type,misc]
    """Table used to save values."""

    # ATTENTION:
    # Prior to modifying this table, please determine whether
    # we should create migrations for this table to make sure
    # users do not experience data loss.
    __tablename__ = "ai_core.langchain_key_value_stores"

    namespace: Mapped[str] = mapped_column(String(64), primary_key=True, nullable=False)
    key: Mapped[str] = mapped_column(String(255), primary_key=True, nullable=False)
    value = mapped_column(Text, index=False, nullable=False)


# This is a fix of original SQLStore.
class SQLBaseStore(BaseStore[str, V], Generic[V]):
    """BaseStore interface that works on an SQL database.

    Examples:
        Create a SQLDocStore instance and perform operations on it:

        .. code-block:: python

            from langchain_rag.storage import SQLDocStore

            # Instantiate the SQLDocStore with the root path
            sql_store = SQLDocStore(namespace="test", db_url="sqllite://:memory:")

            # Set values for keys
            sql_store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values for keys
            values = sql_store.mget(["key1", "key2"])  # Returns [b"value1", b"value2"]

            # Delete keys
            sql_store.mdelete(["key1"])

            # Iterate over keys
            for key in sql_store.yield_keys():
                print(key)

    """

    def __init__(
        self,
        *,
        namespace: str,
        db_url: Optional[Union[str, Path]] = None,
        engine: Optional[Union[Engine, AsyncEngine]] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        async_mode: Optional[bool] = None,
    ):
        if db_url is None and engine is None:
            raise ValueError("Must specify either db_url or engine")

        if db_url is not None and engine is not None:
            raise ValueError("Must specify either db_url or engine, not both")

        _engine: Union[Engine, AsyncEngine]
        if db_url:
            if async_mode is None:
                async_mode = False
            if async_mode:
                _engine = create_async_engine(
                    url=str(db_url),
                    **(engine_kwargs or {}),
                )
            else:
                _engine = create_engine(url=str(db_url), **(engine_kwargs or {}))
        elif engine:
            _engine = engine

        else:
            raise AssertionError("Something went wrong with configuration of engine.")

        _session_maker: Union[sessionmaker[Session], async_sessionmaker[AsyncSession]]
        if isinstance(_engine, AsyncEngine):
            self.async_mode = True
            _session_maker = async_sessionmaker(bind=_engine)
        else:
            self.async_mode = False
            _session_maker = sessionmaker(bind=_engine)

        self.engine = _engine
        self.dialect = _engine.dialect.name
        self.session_maker = _session_maker
        self.namespace = namespace
        self._async_init = False
        if not self.async_mode:
            self.__post_init__()

    def __post_init__(
        self,
    ) -> None:
        """Initialize the store."""
        self.__create_schema()

    async def __apost_init__(
        self,
    ) -> None:
        """Async initialize the store (use lazy approach)."""
        if self._async_init:  # Warning: possible race condition
            return
        self._async_init = True

        await self.__acreate_schema()

    def __create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    async def __acreate_schema(self) -> None:
        assert isinstance(self.engine, AsyncEngine)
        async with self.engine.begin() as session:
            await session.run_sync(Base.metadata.create_all)

    def __serialize_value(self, obj: V) -> str:
        if isinstance(obj, Serializable):
            return dumps(obj)
        if isinstance(obj, str):
            return obj
        raise ValueError(f"Unsupported type: {type(obj)}")

    def __deserialize_value(self, obj: str) -> V:
        try:
            if isinstance(obj, self.__orig_class__.__args__[0]):  # type: ignore
                return obj  # type: ignore
            else:
                return loads(obj)
        except Exception:
            raise

    def drop(self) -> None:
        Base.metadata.drop_all(bind=self.engine.connect())

    async def amget(self, keys: Sequence[str]) -> List[Optional[V]]:
        assert isinstance(self.engine, AsyncEngine)
        await self.__apost_init__()  # Lazy async init

        result: Dict[str, V] = {}
        async with self._make_async_session() as session:
            stmt = select(LangchainKeyValueStores).filter(
                and_(
                    LangchainKeyValueStores.key.in_(keys),
                    LangchainKeyValueStores.namespace == self.namespace,
                )
            )
            for v in await session.scalars(stmt):
                val = self.__deserialize_value(v.value) if v is not None else v
                result[v.key] = val
        return [result.get(key) for key in keys]

    def mget(self, keys: Sequence[str]) -> List[Optional[V]]:
        result: Dict[str, V] = {}

        with self._make_sync_session() as session:
            stmt = select(LangchainKeyValueStores).filter(
                and_(
                    LangchainKeyValueStores.key.in_(keys),
                    LangchainKeyValueStores.namespace == self.namespace,
                )
            )
            for v in session.scalars(stmt):
                val = self.__deserialize_value(v.value) if v is not None else v
                result[v.key] = val
        return [result.get(key) for key in keys]

    async def amset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        await self.__apost_init__()  # Lazy async init
        async with self._make_async_session() as session:
            await self._amdelete([key for key, _ in key_value_pairs], session)
            for key, item in key_value_pairs:
                content = self.__serialize_value(item)
                session.add(
                    LangchainKeyValueStores(
                        namespace=self.namespace, key=key, value=content
                    )
                )
            await session.commit()

    def mset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        with self._make_sync_session() as session:
            self._mdelete([key for key, _ in key_value_pairs], session)
            for key, item in key_value_pairs:
                content = self.__serialize_value(item)
                session.add(
                    LangchainKeyValueStores(
                        namespace=self.namespace, key=key, value=content
                    )
                )
            session.commit()

    def _mdelete(self, keys: Sequence[str], session: Session) -> None:
        stmt = delete(LangchainKeyValueStores).filter(
            and_(
                LangchainKeyValueStores.key.in_(keys),
                LangchainKeyValueStores.namespace == self.namespace,
            )
        )
        session.execute(stmt)

    async def _amdelete(self, keys: Sequence[str], session: AsyncSession) -> None:
        stmt = delete(LangchainKeyValueStores).filter(
            and_(
                LangchainKeyValueStores.key.in_(keys),
                LangchainKeyValueStores.namespace == self.namespace,
            )
        )
        await session.execute(stmt)

    def mdelete(self, keys: Sequence[str]) -> None:
        with self._make_sync_session() as session:
            self._mdelete(keys, session)
            session.commit()

    async def amdelete(self, keys: Sequence[str]) -> None:
        await self.__apost_init__()  # Lazy async init
        async with self._make_async_session() as session:
            await self._amdelete(keys, session)
            await session.commit()

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        with self._make_sync_session() as session:
            stmt = select(LangchainKeyValueStores.key).filter(
                LangchainKeyValueStores.namespace == self.namespace
            )
            if prefix:
                stmt = stmt.filter(LangchainKeyValueStores.key.startswith(prefix))

            for item in session.scalars(stmt):
                yield item
            session.close()

    async def ayield_keys(self, *, prefix: Optional[str] = None) -> AsyncIterator[str]:
        await self.__apost_init__()  # Lazy async init
        async with self._make_async_session() as session:
            stmt = select(LangchainKeyValueStores.key).filter(
                LangchainKeyValueStores.namespace == self.namespace
            )
            if prefix:
                stmt = stmt.filter(LangchainKeyValueStores.key.startswith(prefix))

            for v in await session.scalars(stmt):
                yield v
            await session.close()

    @contextlib.contextmanager
    def _make_sync_session(self) -> Generator[Session, None, None]:
        """Make an async session."""
        if self.async_mode:
            raise ValueError(
                "Attempting to use a sync method in when async mode is turned on. "
                "Please use the corresponding async method instead."
            )
        with cast(Session, self.session_maker()) as session:
            yield cast(Session, session)

    @contextlib.asynccontextmanager
    async def _make_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Make an async session."""
        if not self.async_mode:
            raise ValueError(
                "Attempting to use an async method in when sync mode is turned on. "
                "Please use the corresponding async method instead."
            )
        async with cast(AsyncSession, self.session_maker()) as session:
            yield cast(AsyncSession, session)


SQLDocStore = SQLBaseStore[Document]
SQLStrStore = SQLBaseStore[str]
