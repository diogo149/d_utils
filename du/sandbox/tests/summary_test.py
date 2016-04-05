import nose.tools as nt
from du.sandbox import summary


def test_to_org_list():
    s = summary.Summary()
    s.add("foo")
    s.add("bar")
    s.update({"foo": 3, "bar": 5})
    nt.assert_equal("- foo: 3\n- bar: 5",
                    s.to_org_list())


def test_s_per_iter():
    s = summary.Summary()
    s.add_recipe("s_per_iter")
    nt.assert_equal(0.0, s.value("s_per_iter"))
    s.update({"_time": 3.5, "_iter": 12})
    nt.assert_equal(3.5 / 12, s.value("s_per_iter"))


def test_ms_per_obs():
    s = summary.Summary()
    s.add_recipe("s_per_iter")
    s.add_recipe("ms_per_obs", 100)
    nt.assert_equal(0.0, s.value("ms_per_obs"))
    s.update({"_time": 3.5, "_iter": 12})
    nt.assert_equal(3.5 / 12 * 1000 / 100, s.value("ms_per_obs"))


def test_field_iter():
    s = summary.Summary()
    s.add("foo", how="max")
    s.add_recipe("iter", "foo")
    nt.assert_equal(-1, s.value("foo iter"))
    s.update({"foo": 2, "_iter": 3})
    nt.assert_equal(3, s.value("foo iter"))
    s.update({"foo": 2, "_iter": 10})
    nt.assert_equal(3, s.value("foo iter"))
    s.update({"foo": 3, "_iter": 11})
    nt.assert_equal(11, s.value("foo iter"))


def test_x_max_and_iter():
    s = summary.Summary()
    s.add_recipe("x max+iter", "foo")
    nt.assert_equal(None, s.value("foo max"))
    nt.assert_equal(-1, s.value("foo max iter"))
    s.update({"foo": 2, "_iter": 3})
    nt.assert_equal(2, s.value("foo max"))
    nt.assert_equal(3, s.value("foo max iter"))
    s.update({"foo": 2, "_iter": 10})
    nt.assert_equal(2, s.value("foo max"))
    nt.assert_equal(3, s.value("foo max iter"))
    s.update({"foo": 3, "_iter": 11})
    nt.assert_equal(3, s.value("foo max"))
    nt.assert_equal(11, s.value("foo max iter"))


def test_before_x_minutes():
    s = summary.Summary()
    s.add("foo", how="max")
    s.add_recipe("before_x_minutes", "foo", 2)
    nt.assert_equal(None, s.value("foo before 2 minutes"))
    s.update({"foo": 2, "_time": 3})
    nt.assert_equal(2, s.value("foo"))
    nt.assert_equal(2, s.value("foo before 2 minutes"))
    s.update({"foo": 10, "_time": 10})
    nt.assert_equal(10, s.value("foo"))
    nt.assert_equal(10, s.value("foo before 2 minutes"))
    s.update({"foo": 12, "_time": 120})
    nt.assert_equal(12, s.value("foo"))
    nt.assert_equal(10, s.value("foo before 2 minutes"))
