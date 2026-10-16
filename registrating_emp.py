import sys
from dataclasses import field
from logging import warning
from typing import Literal
import json
from pydantic import BaseModel, ValidationError, Field
import logging



class Registration(BaseModel):
    registration_id : str
    name : str
    email : str
    age : int = Field(ge=18)
    ticket_type : Literal["Standard", "Premium", "VIP"]
    amount : float = Field(gt=0)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            filename="registration.log",
            encoding="utf-8",
        ),
        logging.StreamHandler()
    ],
    force=True
)

valid_registrations =[]
invalid_count = 0

try:
    with open("registrations.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    for record in records:
        try:
            registration = Registration(**record)
            logging.info(f"Valid Registration for {registration.registration_id}")
            valid_registrations.append(registration)

            if registration.ticket_type == "VIP":
                logging.warning(f"VIP registration for {registration.registration_id}")

        except ValidationError as e:
            invalid_count += 1
            logging.error(
                "Registration %s failed validation",
                record.get("registration_id", "UNKNOWN")
            )

            for error in e.errors():
                logging.error(
                    "Field %s: %s",
                    error["loc"][0],
                    error["msg"]
                )

    valid_data = [registration.model_dump() for registration in valid_registrations]

    with open("valid_registrations.json", "w", encoding="utf-8") as f:
        json.dump(valid_data, f, indent=4)

    print("total records: ", len(records))
    print("invalid records: ", invalid_count)
    print("valid records: ", len(valid_registrations))

except FileNotFoundError as e:
    print("Error: registrations.json not found")

except json.decoder.JSONDecodeError as e:
    print("Error: invalid json")

