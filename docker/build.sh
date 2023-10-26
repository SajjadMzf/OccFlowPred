#!/bin/bash
docker build --no-cache ./docker \
             -f docker/Dockerfile \
             -t x64/ofmpnet:latest
              