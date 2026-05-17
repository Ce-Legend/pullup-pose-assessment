from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar

from flask import abort, g, jsonify, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

T = TypeVar("T")


@dataclass(frozen=True)
class User:
    id: int
    username: str
    display_name: str
    role: str


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    return check_password_hash(password_hash, password)


def load_current_user() -> User | None:
    user_dict = session.get("user")
    if not user_dict:
        return None
    return User(
        id=int(user_dict["id"]),
        username=str(user_dict["username"]),
        display_name=str(user_dict["display_name"]),
        role=str(user_dict["role"]),
    )


def login_user(user: User) -> None:
    session["user"] = {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
    }


def logout_user() -> None:
    session.pop("user", None)


def require_login_page(view_func: Callable[..., T]) -> Callable[..., T]:
    @wraps(view_func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        user = load_current_user()
        if user is None:
            return redirect(url_for("login", next=request.path))  # type: ignore[return-value]
        g.user = user
        return view_func(*args, **kwargs)

    return wrapper


def require_login_api(view_func: Callable[..., T]) -> Callable[..., T]:
    @wraps(view_func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        user = load_current_user()
        if user is None:
            return jsonify({"error_code": "UNAUTHORIZED", "error_message": "请先登录"}), 401  # type: ignore[return-value]
        g.user = user
        return view_func(*args, **kwargs)

    return wrapper


def require_teacher_api(view_func: Callable[..., T]) -> Callable[..., T]:
    @wraps(view_func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        user = load_current_user()
        if user is None:
            return jsonify({"error_code": "UNAUTHORIZED", "error_message": "请先登录"}), 401  # type: ignore[return-value]
        if user.role != "teacher":
            abort(403)
        g.user = user
        return view_func(*args, **kwargs)

    return wrapper

