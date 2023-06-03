document.getElementById('fileInput').addEventListener('change', function(e) {
    var fullPath = this.value;
    var fileName = fullPath.split(/(\\|\/)/g).pop();  // Extract the file name from the full path
    document.getElementById('fileName').innerHTML = fileName;
});