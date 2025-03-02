from collections import UserList, namedtuple
import json
from uuid import UUID
from datetime import datetime, date
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import logging
import pytz


class CircularList(UserList):
    __slots__ = ("index",)

    def __init__(self, items):
        super().__init__(items)
        self.index_: int = 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        >>> cars = ["VOLVO", "BMW", "AUDI"]
        >>> my_list = CircularList(cars)
        >>> next(my_list)
        'BMW'
        >>> next(my_list)
        'AUDI'
        >>> next(my_list)
        'VOLVO'
        >>> empty_list = CircularList([])
        >>> next(empty_list)
        
        """
        if not self.data:
            return None
        self.index_ = (self.index_ + 1) % len(self.data)
        return self.data[self.index_]

    @property
    def current(self):
        return self.data[self.index_]

    def reset_index(self):
        self.index_ = 0
        return self.data[self.index_]


def dict_to_text(d, indent=0):
    """
    Convert a dictionary to a formatted text representation with indentation.

    >>> d = {
    ...     "hello": {
    ...         "world": "man"
    ...     }
    ... }
    >>> print(dict_to_text(d, indent=4))
        ▸ hello:
            ▸ world: man
    """
    text = ""
    for key, value in d.items():
        if isinstance(value, dict):
            text += " " * indent + f"▸ {key}:\n"
            text += dict_to_text(value, indent + 4)
        else:
            text += " " * indent + f"▸ {key}: {value}\n"
    return text.rstrip()


class Encoder(json.JSONEncoder):
    """
    A custom JSON encoder that handles datetime, date, and UUID objects.

    Examples:
        >>> test_uuid = UUID('d736034e-2239-4fc6-a1c9-4ba9df20888f')
        >>> json.dumps({
        ...     "timestamp": datetime(2023, 10, 5, 12, 30, 45),
        ...     "test_uuid": test_uuid,
        ... }, cls=Encoder)
        '{"timestamp": "2023-10-05 12:30:45", "test_uuid": "d736034e-2239-4fc6-a1c9-4ba9df20888f"}'
    """
    def default(self, o):
        if isinstance(o, (datetime, date, UUID)):
            return str(o)
        return super().default(o)


Range = namedtuple("Range", ["min_", "max_"])

def get_range(field: str | None) -> Range:
    """
    Parse a string field into a Range namedtuple.

    Examples:
        >>> get_range(None)  # Test with None input
        Range(min_=None, max_=None)

        >>> get_range("1,10")  # Test with valid range
        Range(min_=1, max_=10)

        >>> get_range("5")  # Test with single value
        Range(min_=5, max_=None)

        >>> get_range("1,")  # Test with missing max value
        Range(min_=1, max_=None)

        >>> get_range(",10")  # Test with missing min value
        Range(min_=None, max_=10)

        >>> get_range(",")  # Test with empty values
        Range(min_=None, max_=None)

        >>> get_range("")  # Test with empty string
        Range(min_=None, max_=None)
    """
    if not field:
        return Range(None, None)

    range_arr = field.split(",")

    if len(range_arr) < 2:
        range_arr.extend([None] * (2 - len(range_arr)))

    range_arr = [
        int(value) if value is not None and value.strip() != "" else None
        for value in range_arr
    ]

    return Range(range_arr[0], range_arr[1])

def replace_key_recursively(data, target_key, new_value):
    """
    Recursively replaces all values of `target_key` with `new_value` in a nested dictionary or list structure.

    Examples:
        >>> data = {"a": 1, "b": {"a": 2, "c": 3}, "d": [{"a": 4}, {"b": 5}]}
        >>> replace_key_recursively(data, "a", 99)
        {'a': 99, 'b': {'a': 99, 'c': 3}, 'd': [{'a': 99}, {'b': 5}]}

        >>> data = [{"x": 1}, {"y": {"x": 2}}]
        >>> replace_key_recursively(data, "x", 100)
        [{'x': 100}, {'y': {'x': 100}}]

        >>> data = {"key1": "value1", "key2": "value2"}
        >>> replace_key_recursively(data, "key3", "new_value")  # Key not present
        {'key1': 'value1', 'key2': 'value2'}

        >>> data = {}
        >>> replace_key_recursively(data, "key", "value")  # Empty input
        {}

        >>> data = {"a": {"b": {"a": 1}}}
        >>> replace_key_recursively(data, "a", "replaced")
        {'a': 'replaced'}
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                data[key] = new_value
            if isinstance(value, (dict, list)):
                replace_key_recursively(value, target_key, new_value)
    elif isinstance(data, list):
        for item in data:
            replace_key_recursively(item, target_key, new_value)
    return data


def split_array(arr: list, chunk_size: int) -> list[list]:
    """
    Splits an array into multiple subarrays of a specified size.

    :param arr: The input array to be split.
    :param chunk_size: The size of each chunk.
    :return: A list of subarrays, each of size `chunk_size` (except possibly the last one).

    Examples:
        >>> split_array([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]

        >>> split_array([1, 2, 3, 4, 5], 3)
        [[1, 2, 3], [4, 5]]

        >>> split_array([1, 2, 3, 4, 5], 5)
        [[1, 2, 3, 4, 5]]

        >>> split_array([1, 2, 3, 4, 5], 1)
        [[1], [2], [3], [4], [5]]

        >>> split_array([], 2)  # Empty array
        []

        >>> split_array([1, 2, 3, 4, 5], 10)  # Chunk size larger than array
        [[1, 2, 3, 4, 5]]
    """
    return [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]

def convert_utc_to_local(
    datetime_str_utc: str | datetime, timezone_str: str, only_time: bool = True
) -> str:
    """
    Convert a UTC datetime to a local datetime based on the given UTC offset.

    Args:
        datetime_str_utc (str | datetime): The UTC datetime as a string (in ISO format) or a datetime object.
        timezone_str (str): The timezone offset in the format "UTC±X" (e.g., "UTC+3", "UTC-5", "UTC+2.5").
        only_time (bool): If True, return only the time part. If False, return the full datetime.

    Returns:
        str: The local datetime or time as a formatted string.

    Examples:
        >>> convert_utc_to_local("2023-10-01T12:00:00", "UTC+3")
        '03:00 PM'

        >>> convert_utc_to_local("2023-10-01T12:00:00", "UTC-5")
        '07:00 AM'

        >>> convert_utc_to_local("2023-10-01T12:00:00", "UTC+2.5", only_time=False)
        '2023-10-01 02:30 PM'

        >>> convert_utc_to_local("2023-10-01T12:00:00", "UTC+0")
        '12:00 PM'
    """
    if isinstance(datetime_str_utc, str):
        datetime_utc = datetime.fromisoformat(datetime_str_utc)
    else:
        datetime_utc = datetime_str_utc

    match_timezone = re.match(r"UTC([+-]?\d+(\.\d+)?)", timezone_str.upper())
    if not match_timezone:
        raise ValueError("Invalid timezone format. Expected format: UTC±X.")

    timezone_offset = float(match_timezone.group(1))
    offset_minutes = int(timezone_offset * 60)
    timezone = pytz.FixedOffset(offset_minutes)

    datetime_with_timezone = datetime_utc.replace(tzinfo=pytz.UTC).astimezone(timezone)

    if only_time:
        return datetime_with_timezone.strftime("%I:%M %p")
    return datetime_with_timezone.strftime("%Y-%m-%d %I:%M %p")


class ScriptWatcher:
    """
    A class to watch for changes in Python files and restart a script automatically.

    Args:
        script_path (str): The path to the script to be executed (e.g., "bot.py").
        watch_path (str): The directory path to watch for changes (default is current directory ".").

    Examples:
      if __name__ == "__main__":
        watcher = ScriptWatcher(script_path="bot.py", watch_path=".")
        try:
            watcher.start()
            while True:
                time.sleep(4)
        except KeyboardInterrupt:
            watcher.stop()
    """

    def __init__(self, script_path: str, watch_path: str = "."):
        self.script_path = script_path
        self.watch_path = watch_path
        self.observer = Observer()
        self.event_handler = self._create_event_handler()
        self.process = None
        self.logger = logging.getLogger("watcher")

    def _create_event_handler(self):
        """Create a custom event handler for file system events."""

        class PythonFileHandler(FileSystemEventHandler):
            def __init__(self, restart_callback):
                self.restart_callback = restart_callback
                self.logger = logging.getLogger("watcher_handler")

            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith(".py"):
                    self.logger.info(f"Detected changes in file: {event.src_path}")
                    self.restart_callback()

        return PythonFileHandler(restart_callback=self.restart_script)

    def restart_script(self):
        """Restart the script by terminating the current process and starting a new one."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
        self.process = subprocess.Popen(["python3", self.script_path])

    def start(self):
        """Start watching for file changes and execute the script."""
        self.observer.schedule(self.event_handler, path=self.watch_path, recursive=True)
        self.observer.start()
        self.restart_script()  # Start the script initially
        self.logger.info(f"Watching for changes in '{self.watch_path}' and running '{self.script_path}'...")

    def stop(self):
        """Stop watching for file changes and terminate the script."""
        self.observer.stop()
        self.observer.join()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
        self.logger.info("Stopped watching for changes.")

from typing import Type, TypeVar, Generic
from pydantic import BaseModel
import json

ConfigModel = TypeVar("ConfigModel", bound=BaseModel)


class ConfigLoader(Generic[ConfigModel]):
    """
    A class to load and validate configuration settings from a JSON file.

    Attributes:
        config_path (str): Path to the JSON configuration file.
        model_class (Type[ConfigModel]): The Pydantic model class used for validation.
        raw_config (dict): The raw configuration data loaded from the JSON file.

    Methods:
        load_config(): Loads the configuration from the JSON file.
        model: Property that returns a validated Pydantic model instance.
    """

    def __init__(self, model_class: Type[ConfigModel], config_path: str = " ") -> None:
        """
        Initializes the ConfigLoader with the specified model class and configuration file path.

        Args:
            model_class (Type[ConfigModel]): The Pydantic model class to use for validation.
            config_path (str): Path to the JSON configuration file. Defaults to " ".
        """
        self.model_class = model_class
        self.config_path = config_path
        self.raw_config = self.load_config()

        for key, value in self.raw_config.items():
            setattr(self, key, value)

    def load_config(self) -> dict:
        """
        Loads the configuration from the JSON file specified by config_path.

        Returns:
            dict: The configuration data.

        Raises:
            ValueError: If the configuration file is empty or not found.
        """
        with open(self.config_path, "rb") as file:
            data = json.load(file)

        if not data:
            raise ValueError("Configuration file is empty or not found.")
        return data

    @property
    def model(self) -> ConfigModel:
        """
        Validates the raw configuration data against the specified Pydantic model.

        Returns:
            ConfigModel: A validated instance of the Pydantic model.

        Raises:
            ValueError: If the model class is not set.
            TypeError: If the model class is not a subclass of BaseModel.
        """
        if not issubclass(self.model_class, BaseModel):
            raise TypeError("The model class must be a subclass of BaseModel.")

        return self.model_class.model_validate(self.raw_config)


class SingletonConfigLoader(ConfigLoader, Generic[ConfigModel]):
    """
    A singleton version of ConfigLoader to ensure only one instance exists.

    Attributes:
        _instance (SingletonConfigLoader): The single instance of the class.
        _initialized (bool): Flag to check if the instance has been initialized.
        _model (ConfigModel | None): Cached validated model instance.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of SingletonConfigLoader is created.

        Returns:
            SingletonConfigLoader: The single instance of the class.
        """
        if not isinstance(cls._instance, cls):
            cls._instance = super(SingletonConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_class: Type[ConfigModel] | None = None, config_path: str = " "):
        """
        Initializes the SingletonConfigLoader with the specified model class and configuration file path.

        Args:
            model_class (Type[ConfigModel] | None): The Pydantic model class to use for validation.
            config_path (str): Path to the JSON configuration file. Defaults to " ".
        """
        if not getattr(self, "_initialized", False):
            if model_class is None:
                raise ValueError("Model class must be provided for the first initialization.")
            super().__init__(model_class, config_path)
            self._initialized = True
            self._model = None

    @property
    def model(self) -> ConfigModel:
        """
        Validates the raw configuration data against the specified Pydantic model and caches the result.

        Returns:
            ConfigModel: A validated instance of the Pydantic model.

        Raises:
            ValueError: If the model class is not set.
            TypeError: If the model class is not a subclass of BaseModel.
        """
        if self._model:
            return self._model

        if not issubclass(self.model_class, BaseModel):
            raise TypeError("The model class must be a subclass of BaseModel.")

        self._model = self.model_class.model_validate(self.raw_config)
        return self._model


# Example usage with doctest
if __name__ == "__main__":
    import doctest

    class SampleConfig(BaseModel):
        name: str
        value: int

    def test_config_loader():
        """
        Test the ConfigLoader class.

        >>> loader = ConfigLoader(SampleConfig, "sample_config.json")
        >>> config = loader.model
        >>> isinstance(config, SampleConfig)
        True
        >>> config.name
        'example'
        >>> config.value
        42
        """

    def test_singleton_config_loader():
        """
        Test the SingletonConfigLoader class.

        >>> loader1 = SingletonConfigLoader(SampleConfig, "sample_config.json")
        >>> config1 = loader1.model
        >>> loader2 = SingletonConfigLoader()
        >>> config2 = loader2.model
        >>> config1 is config2
        True
        """

    with open("sample_config.json", "w") as f:
        json.dump({"name": "example", "value": 42}, f)

    doctest.testmod()

    import os
    os.remove("sample_config.json")