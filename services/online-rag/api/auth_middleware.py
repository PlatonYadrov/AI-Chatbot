from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import Depends, HTTPException, Request, status


def _parse_roles(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [r.strip() for r in raw.split(",") if r.strip()]


def get_user_ctx(request: Request) -> Dict[str, object]:
    """
    Заглушка аутентификации (OIDC/JWT позже):
    - Берёт user_id/roles/department/manager_type из заголовков.
    - deny-by-default: 401 если user_id отсутствует.
    """
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing user id")
    roles = _parse_roles(request.headers.get("x-roles"))
    dept = request.headers.get("x-dept")
    manager_type = request.headers.get("x-manager-type") or "line"
    return {"id": user_id, "dept": dept, "roles": roles, "manager_type": manager_type}


UserContext = Dict[str, object]


__all__ = ["get_user_ctx", "UserContext"]

