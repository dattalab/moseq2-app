#!/bin/bash
mkdir test_data
cd test_data

# amphetamine data
curl -L -o amphetamine_example_0.zip https://www.dropbox.com/sh/83fqdwq286g8s1v/AABi4UJ_qUz4NCO11eWxKSDqa?dl=1 && unzip amphetamine_example_0.zip -d amphetamine_example_0
rm amphetamine_example_0.zip

# saline data
curl -L -o saline_example_0.zip https://www.dropbox.com/sh/llp16zjf0jrsq0w/AABbNoYMOOdgqCWYSbR4Upn-a?dl=1 && unzip saline_example_0.zip -d saline_example_0
rm saline_example_0.zip