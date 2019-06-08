<?php

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| contains the "web" middleware group. Now create something great!
|
*/

/*Route::get('/', function () {
    return view('welcome');
});*/

Route::get('/','HomeController@index');

Route::get('media/upload_image', ['uses' => 'MediaController@index'])->name('media.upload_image');
Route::post('media/upload_image', ['uses' => 'MediaController@uploadImage'])->name('media.upload_image');
