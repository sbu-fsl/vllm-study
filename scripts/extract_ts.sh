#!/bin/bash

FILE=$1

echo "START: "
grep "vLLM API server version" "${FILE}"

echo "END: "
grep -B 1 "Started server process" "${FILE}" | grep -v "Started server process"
