#!/usr/bin/env bash
set -euo pipefail
CAMPAIGN=/home/User/rsna-resnet50-20261013
PACKAGE_SHA=9ee1581ecd3ce6bf0bf9972ce82ee2bf3fe810d43eabc3dbfb3ee1bca823a77f
STATUS="$CAMPAIGN/deployment_status.json"
failures=0
while true; do
    if [[ -f "$STATUS" ]] && grep -q 'training_complete' "$STATUS"; then
        exit 0
    fi
    if flock -n "$CAMPAIGN/bootstrap.lock" -c true; then
        if bash /home/User/bootstrap_resnet50_cloud.sh "$PACKAGE_SHA" >> "$CAMPAIGN/watchdog.log" 2>&1; then
            if [[ -f "$STATUS" ]] && grep -q 'training_complete' "$STATUS"; then
                exit 0
            fi
        fi
        failures=$((failures + 1))
        if (( failures >= 3 )); then
            printf 'Stopped after %s failed relaunches\n' "$failures" >> "$CAMPAIGN/watchdog.log"
            exit 1
        fi
        sleep 120
    else
        sleep 60
    fi
done
