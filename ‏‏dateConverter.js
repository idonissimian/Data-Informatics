window.onload = function () {
    // Click on submit
    $(document).ready(function(){
        $("#gregorianDateSubmit").click(function(){
            // Get the date input values
            var dateValue = document.querySelector('#gregorianDateInput').value;
            if(dateValue != "")
            {
                // Get the day, the month and the year
                var year = dateValue.substring(0, 4);
                var month = dateValue.substring(5, 7);
                var day = dateValue.substring(8, 10);
                var note = '('+day+'/'+month+'/'+year+' : עבור התאריך הלועזי)';
                url = 'https://www.hebcal.com/converter?cfg=json&gy='+year+'&gm='+month+'&gd='+day+'&g2h=1'
                // Get the json file from the date input
                $.getJSON(url, function(data) {
                    // Get the hebrew date from the JSON file
                    var hebrewDate = data.hebrew;
                    // Show the hebrew date
                    document.getElementById("hebrewDateShow").innerHTML = hebrewDate;
                    document.getElementById("hebrewDateNote").innerHTML = note;
                    document.getElementById("hebrewDateSection").style.display = "block";
                });
            }
        });
    });
};