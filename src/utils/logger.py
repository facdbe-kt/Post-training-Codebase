import logging, yaml
from logging.handlers import RotatingFileHandler
import os
from typing import List

from .config import Config
from .funcs import save_json, copy_file_or_directory

class Logger:
    def __init__(self, name, level=logging.DEBUG, cfg=None):
        self.cfg = Config.load_current_config() if cfg is None else cfg
        self.debug = self.cfg.get('debug')
        self.line_limit = self.cfg.get('line_limit')

        # parse the name: [FileName] LoggerName
        if name.startswith('[') and ']' in name:
            logger_name = name.split(']')[-1].strip()
            filename = name.split(']')[0].strip()[1:]
        else:
            logger_name = name
            filename = name

        """Initialize the logger with a specific name and level."""
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        self.log_file =  f'{filename}.md'
        self.root = os.path.join(self.cfg.get('filecachedir'), f"versions", self.cfg.get_version(), 'logs')
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.FIRST_N_LINES = self.cfg.get('cut_first_n_line')
        self.LAST_N_LINES = self.cfg.get('cut_last_n_line')
        self.MAX_LINES = self.FIRST_N_LINES + self.LAST_N_LINES

        os.makedirs(self.root, exist_ok=True)
        self.write_first_line_to_log_file()

        self.formatter = logging.Formatter('\n# ------------[%(name)s](%(levelname)s)-%(asctime)s------------\n\n%(message)s')
        # Check if handlers are already configured
        if not self.logger.handlers: 
            # Create file handler for rotating logs
            file_handler = RotatingFileHandler(os.path.join(self.root, self.log_file), maxBytes=1024*1024*50, backupCount=10, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(self.formatter)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(self.formatter)

            # Add both handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    @staticmethod
    def save_args_files(cfg:Config=None, arg_dict:dict=None, files_to_save:List[str]=None, other_infos:dict=None):
        cfg = Config.load_current_config() if cfg is None else cfg
        save_root = os.path.join(cfg.get('filecachedir'), f"versions", cfg.get_version(), 'trials_args')
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        
        # save the config
        cfg_path = os.path.join(save_root, f'_CONFIG_{cfg.get("name")}.yaml')
        with open(cfg_path, "w", encoding='utf-8') as f:
            yaml.dump(cfg, f)
        # save the args
        if arg_dict is not None:
            for arg_name, arg in arg_dict.items():
                arg_path = os.path.join(save_root, f'_ARG_{arg_name}.yaml')
                with open(arg_path, "w", encoding='utf-8') as f:
                    yaml.dump(arg, f)
        # save the files
        if files_to_save is not None:
            file_save_root = os.path.join(save_root, 'file_backup')
            for file_path in files_to_save:
                copy_file_or_directory(file_path, file_save_root)

        # save the other infos
        if other_infos is not None:
            # save as json
            save_json(other_infos, os.path.join(save_root, f'_OTHER_INFOS.json'))

    def write_first_line_to_log_file(self, first_line:str="# LOG CONTENT"):
        """Write the first line to the log file."""
        if not os.path.exists(os.path.join(self.root, self.log_file)):
            with open(os.path.join(self.root, self.log_file), 'a', encoding='utf-8') as f:
                f.write(first_line + '\n')
    
    def get_logger(self):
        """Return the configured logger."""
        return self.logger
    
    def cut_last_tbl(self, msg):
        
        lines = msg.split('\n')
        last_beg, last_end = -1, -1
        for i, line in enumerate(lines):
            if line.strip() == '/*':
                last_beg = i
            if line.strip() == '*/':
                last_end = i
        
        if last_beg >= last_end:
            return msg
        
        # cut the table, only keep the first 1 lines and last 1 lines
        new_prompt = '\n'.join(lines[:last_beg+4] + ['......'] + lines[last_end-1:])

        return new_prompt
    
    def log(self, *values: object, sep: str = " ", level='debug'):
        """Log the values with the specified level."""
        msg = sep.join(map(str, values))

        if not self.debug:
            return
        lines = str(msg).split('\n')    
        if self.line_limit and self.MAX_LINES>0 and len(lines) > self.MAX_LINES:
            msg = self.cut_last_tbl(msg = str(msg))
            lines = msg.split('\n')    
            lines = lines[:self.FIRST_N_LINES] + ['......'] + lines[-self.LAST_N_LINES:]
            msg = '\n'.join(lines)
        
        if level == 'debug':
            self.logger.debug(msg=msg)
        elif level == 'info':
            self.logger.info(msg)
        elif level == 'warning':
            self.logger.warning(msg)
        elif level == 'error':
            self.logger.error(msg=msg)
        elif level == 'critical':
            self.logger.critical(msg)