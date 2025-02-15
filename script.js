document.getElementById("predictBtn").addEventListener("click", async function () {
    const formData = {
        age: document.getElementById("age").value,
        gender: document.getElementById("gender").value,
        income: document.getElementById("income").value,
        health_status: document.getElementById("health_status").value,
        smoking: document.getElementById("smoking").value,
        family_history: document.getElementById("family_history").value
    };

    try {
        const response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        });

        const data = await response.json();
        document.getElementById("result").innerHTML = `
            <p><strong>Eligibility:</strong> ${data.eligibility}</p>
            <p><strong>Policies:</strong> ${data.policies.join(", ")}</p>
            <p><strong>Premium Estimates:</strong> ${JSON.stringify(data.premiums)}</p>
        `;
    } catch (error) {
        console.error("Error fetching prediction:", error);
        document.getElementById("result").innerHTML = <p style="color: red;">Error fetching prediction</p>;
    }
});

// 3D Sphere Animation
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const geometry = new THREE.SphereGeometry(1.5, 64, 64);
const material = new THREE.MeshStandardMaterial({ color: "royalblue", roughness: 0.5, metalness: 0.7 });
const sphere = new THREE.Mesh(geometry, material);
scene.add(sphere);

const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(2, 2, 2);
scene.add(light);

camera.position.z = 5;

function animate() {
    requestAnimationFrame(animate);
    sphere.rotation.y += 0.01;
    renderer.render(scene, camera);
}
animate();

// Handle resizing
window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});