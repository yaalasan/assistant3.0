const API_URL = "https://hualiassist.lol"; // hualiassist.lol

async function uploadPDF() {
  const file = document.getElementById("fileInput").files[0];
  if (!file) return alert("Please select a PDF.");

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/upload_pdf`, {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  document.getElementById("status").innerText = data.message;
}

async function askQuestion() {
  const question = document.getElementById("questionInput").value;

  const formData = new FormData();
  formData.append("request", question);

  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  document.getElementById("answer").innerText = data.reply;
}
