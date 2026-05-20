import os
from typing import Any, Mapping


ROLE_ADMIN = "admin"
ROLE_MEDICAL = "medical_reviewer"
ROLE_LEGAL = "legal_reviewer"
ROLE_REGULATORY = "regulatory_reviewer"

ROLE_LABELS = {
    ROLE_ADMIN: "Admin",
    ROLE_MEDICAL: "Medical Reviewer",
    ROLE_LEGAL: "Legal Reviewer",
    ROLE_REGULATORY: "Regulatory Reviewer",
}

ROLE_DEPARTMENTS = {
    ROLE_ADMIN: "Admin",
    ROLE_MEDICAL: "Medical",
    ROLE_LEGAL: "Legal",
    ROLE_REGULATORY: "Regulatory",
}

ROLE_ALIASES = {
    "admin": ROLE_ADMIN,
    "administrator": ROLE_ADMIN,
    "medical": ROLE_MEDICAL,
    "medical reviewer": ROLE_MEDICAL,
    "medical_reviewer": ROLE_MEDICAL,
    "legal": ROLE_LEGAL,
    "legal reviewer": ROLE_LEGAL,
    "legal_reviewer": ROLE_LEGAL,
    "regulatory": ROLE_REGULATORY,
    "regulatory reviewer": ROLE_REGULATORY,
    "regulatory_reviewer": ROLE_REGULATORY,
}

FEATURE_PERMISSIONS = {
    "upload_files": {ROLE_ADMIN, ROLE_MEDICAL, ROLE_LEGAL, ROLE_REGULATORY},
    "view_uploaded_material": {ROLE_ADMIN, ROLE_MEDICAL, ROLE_LEGAL, ROLE_REGULATORY},
    "view_extracted_intelligence": {ROLE_ADMIN, ROLE_MEDICAL, ROLE_LEGAL, ROLE_REGULATORY},
    "view_key_message_mapping": {ROLE_ADMIN, ROLE_MEDICAL, ROLE_LEGAL, ROLE_REGULATORY},
    "view_summary": {ROLE_ADMIN, ROLE_MEDICAL, ROLE_LEGAL, ROLE_REGULATORY},
    "view_mlr_review": {ROLE_ADMIN, ROLE_MEDICAL, ROLE_LEGAL, ROLE_REGULATORY},
    "add_review_comments": {ROLE_ADMIN, ROLE_MEDICAL, ROLE_LEGAL, ROLE_REGULATORY},
    "manage_users_roles": {ROLE_ADMIN},
    "delete_files": {ROLE_ADMIN},
    "delete_comments": {ROLE_ADMIN},
    "change_system_settings": {ROLE_ADMIN},
}


def normalize_role(role: str | None) -> str:
    value = str(role or os.getenv("DEFAULT_USER_ROLE") or ROLE_ADMIN).strip().lower().replace("-", " ")
    return ROLE_ALIASES.get(value, ROLE_ADMIN)


def role_label(role: str | None) -> str:
    return ROLE_LABELS.get(normalize_role(role), ROLE_LABELS[ROLE_ADMIN])


def department_for_role(role: str | None) -> str:
    return ROLE_DEPARTMENTS.get(normalize_role(role), ROLE_DEPARTMENTS[ROLE_ADMIN])


def get_current_user_role(headers: Mapping[str, Any] | None = None) -> str:
    headers = headers or {}
    role = headers.get("x-user-role") or headers.get("X-User-Role") or os.getenv("DEFAULT_USER_ROLE")
    return normalize_role(str(role or ROLE_ADMIN))


def get_current_user_name(headers: Mapping[str, Any] | None = None) -> str:
    headers = headers or {}
    name = headers.get("x-user-name") or headers.get("X-User-Name") or os.getenv("DEFAULT_USER_NAME")
    return str(name or "Local User").strip() or "Local User"


def is_admin(role: str | None) -> bool:
    return normalize_role(role) == ROLE_ADMIN


def can_view(feature: str, role: str | None) -> bool:
    return normalize_role(role) in FEATURE_PERMISSIONS.get(feature, set())


def can_add_comment(role: str | None) -> bool:
    return can_view("add_review_comments", role)


def can_edit_comment(
    role: str | None,
    comment_owner_role: str | None,
    comment_owner_user: str | None,
    current_user: str | None,
    comment_department: str | None = None,
) -> bool:
    role = normalize_role(role)
    if is_admin(role):
        return True
    if department_for_role(role).lower() != str(comment_department or "").strip().lower():
        return False
    if normalize_role(comment_owner_role) != role:
        return False
    return str(comment_owner_user or "").strip().lower() == str(current_user or "").strip().lower()


def can_resolve_comment(role: str | None, comment_department: str | None) -> bool:
    role = normalize_role(role)
    if is_admin(role):
        return True
    return department_for_role(role).lower() == str(comment_department or "").strip().lower()


def can_delete_comment(role: str | None) -> bool:
    return can_view("delete_comments", role)


def can_approve(role: str | None, department: str | None) -> bool:
    role = normalize_role(role)
    if is_admin(role):
        return True
    return department_for_role(role).lower() == str(department or "").strip().lower()


def permissions_for_role(role: str | None) -> dict[str, bool]:
    role = normalize_role(role)
    return {feature: role in allowed_roles for feature, allowed_roles in FEATURE_PERMISSIONS.items()}


def role_options() -> list[dict[str, str]]:
    return [{"value": value, "label": label, "department": ROLE_DEPARTMENTS[value]} for value, label in ROLE_LABELS.items()]
