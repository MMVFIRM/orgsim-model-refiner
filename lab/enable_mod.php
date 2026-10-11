<?php
define('NOCSRFCHECK',1); $sapi_type=php_sapi_name(); 
$_SERVER['DOCUMENT_ROOT']='/var/www/html';
require '/var/www/html/master.inc.php';
require_once '/var/www/html/core/lib/admin.lib.php';
$user->fetch(1); $user->getrights();
foreach (array('modStock') as $m) { $r = activateModule($m); echo $m.': '.json_encode($r)."\n"; }
