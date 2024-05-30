// Description: This file contains the functions that are used in the intro.html file.

function calculateDaysBetweenDates(begin, end) { 
  const millisecondsPerDay = 86400000; 
  const diff = end - begin; 
  return Math.floor(diff / millisecondsPerDay); 
}


