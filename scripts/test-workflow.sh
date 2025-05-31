#!/bin/bash

echo "🧪 Testing GitHub Actions Workflow Fix"
echo "======================================"

# Test 1: YAML syntax validation
echo "1️⃣ Testing YAML syntax..."
if command -v python3 &> /dev/null; then
    python3 -c "
import yaml
import sys
try:
    with open('.github/workflows/smart-deploy.yml', 'r') as file:
        yaml.safe_load(file)
    print('✅ YAML syntax is valid')
except yaml.YAMLError as e:
    print(f'❌ YAML syntax error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error reading file: {e}')
    sys.exit(1)
"
else
    echo "⚠️  Python3 not available for YAML validation"
fi

# Test 2: Check secret references
echo ""
echo "2️⃣ Validating secret references..."
secret_issues=0

echo "Checking for VM_HOST references..."
vm_host_count=$(grep -c "VM_HOST" .github/workflows/smart-deploy.yml)
if [ "$vm_host_count" -gt 0 ]; then
    echo "✅ Found $vm_host_count VM_HOST references"
else
    echo "❌ No VM_HOST references found"
    secret_issues=$((secret_issues + 1))
fi

echo "Checking for VM_SSH_KEY references..."
vm_ssh_key_count=$(grep -c "VM_SSH_KEY" .github/workflows/smart-deploy.yml)
if [ "$vm_ssh_key_count" -gt 0 ]; then
    echo "✅ Found $vm_ssh_key_count VM_SSH_KEY references"
else
    echo "❌ No VM_SSH_KEY references found"
    secret_issues=$((secret_issues + 1))
fi

echo "Checking for old VM_IP references..."
vm_ip_count=$(grep -c "VM_IP" .github/workflows/smart-deploy.yml)
if [ "$vm_ip_count" -eq 0 ]; then
    echo "✅ No old VM_IP references found"
else
    echo "❌ Found $vm_ip_count old VM_IP references - should be VM_HOST"
    secret_issues=$((secret_issues + 1))
fi

# Test 3: SSH action validation
echo ""
echo "3️⃣ Validating SSH action configuration..."
ssh_issues=0

echo "Checking appleboy/ssh-action usage..."
ssh_action_count=$(grep -c "appleboy/ssh-action" .github/workflows/smart-deploy.yml)
if [ "$ssh_action_count" -gt 0 ]; then
    echo "✅ Found $ssh_action_count appleboy/ssh-action usages"
else
    echo "❌ No appleboy/ssh-action found"
    ssh_issues=$((ssh_issues + 1))
fi

echo "Checking SSH action parameters..."
if grep -A 5 "appleboy/ssh-action" .github/workflows/smart-deploy.yml | grep -q "host:" && \
   grep -A 5 "appleboy/ssh-action" .github/workflows/smart-deploy.yml | grep -q "username:" && \
   grep -A 5 "appleboy/ssh-action" .github/workflows/smart-deploy.yml | grep -q "key:"; then
    echo "✅ SSH action has required parameters (host, username, key)"
else
    echo "❌ SSH action missing required parameters"
    ssh_issues=$((ssh_issues + 1))
fi

# Test 4: Deployment logic validation
echo ""
echo "4️⃣ Validating deployment job structure..."
deployment_issues=0

echo "Checking deploy-to-vm job..."
if grep -q "deploy-to-vm:" .github/workflows/smart-deploy.yml; then
    echo "✅ deploy-to-vm job found"
else
    echo "❌ deploy-to-vm job missing"
    deployment_issues=$((deployment_issues + 1))
fi

echo "Checking health-check job..."
if grep -q "health-check:" .github/workflows/smart-deploy.yml; then
    echo "✅ health-check job found"
else
    echo "❌ health-check job missing"
    deployment_issues=$((deployment_issues + 1))
fi

echo "Checking job dependencies..."
if grep -q "needs: \[deploy-to-vm\]" .github/workflows/smart-deploy.yml; then
    echo "✅ health-check depends on deploy-to-vm"
else
    echo "❌ health-check job dependency missing"
    deployment_issues=$((deployment_issues + 1))
fi

# Test 5: Service endpoints validation
echo ""
echo "5️⃣ Validating service endpoints..."
endpoint_issues=0

services=("yolo-combined:8001" "mmpose:8003" "yolo-nas:8004" "rf-detr:8005" "vitpose:8006")
for service in "${services[@]}"; do
    service_name=$(echo "$service" | cut -d: -f1)
    port=$(echo "$service" | cut -d: -f2)
    
    if grep -q "$service_name.*$port" .github/workflows/smart-deploy.yml; then
        echo "✅ $service_name service on port $port found"
    else
        echo "❌ $service_name service on port $port missing"
        endpoint_issues=$((endpoint_issues + 1))
    fi
done

# Summary
echo ""
echo "📊 VALIDATION SUMMARY"
echo "===================="
total_issues=$((secret_issues + ssh_issues + deployment_issues + endpoint_issues))

if [ $total_issues -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED!"
    echo "✅ Workflow is properly configured for deployment"
    echo ""
    echo "🔧 Key fixes verified:"
    echo "  - VM_HOST secrets used correctly"
    echo "  - No old VM_IP references"
    echo "  - SSH actions properly configured"
    echo "  - Deployment jobs properly structured"
    echo "  - All service endpoints validated"
    exit 0
else
    echo "❌ VALIDATION FAILED!"
    echo "Issues found:"
    echo "  - Secret issues: $secret_issues"
    echo "  - SSH action issues: $ssh_issues" 
    echo "  - Deployment issues: $deployment_issues"
    echo "  - Endpoint issues: $endpoint_issues"
    echo "Total issues: $total_issues"
    exit 1
fi