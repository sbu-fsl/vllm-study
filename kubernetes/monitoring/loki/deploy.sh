#!/bin/bash

helm install --values values.yaml loki grafana/loki --namespace loki --create-namespace

