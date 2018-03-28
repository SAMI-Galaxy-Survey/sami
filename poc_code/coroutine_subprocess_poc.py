"""

git-repository: coroutine_subprocess.py

Based on: https://stackoverflow.com/questions/34020599/asynchronously-receive-output-from-long-running-shell-commands-with-asyncio-pyt

History
-------

Created by: agreen on 13/2/18


"""
from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Any

import asyncio

import logging.config

log = logging.getLogger(__name__)
logging_level = logging.WARNING
# Logging configuration at end of file.


import sys
import time
from asyncio.subprocess import PIPE, STDOUT, DEVNULL

async def inner_layer(shell_command):
    p = await asyncio.create_subprocess_shell(shell_command,
            stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    return (await p.communicate())[0].splitlines()


async def get_lines(shell_command):
    res = await inner_layer(shell_command)
    return res

async def main():
    # get commands output concurrently
    coros = [get_lines('"{e}" -c "print({i:d}); import time; time.sleep({i:d})"'
                       .format(i=i+3, e=sys.executable))
             for i in reversed(range(5))]
    # for f in asyncio.as_completed(coros):  # print in the order they finish
    #     print(await f)
    res = await asyncio.gather(*coros)
    for i in res:
        print(i)


if __name__ == "__main__":
    start_time = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

    print("Took total wall time of {} seconds".format(time.time() - start_time))













logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(levelname)s %(filename)s:%(lineno)s %(funcName)s: %(message)s'
        },
        'simple': {
            'format': '%(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose'
        }
    },
    'loggers': {
        'sds': {
            'handlers': ['console'],
            'level': logging_level
        }
    }
})
